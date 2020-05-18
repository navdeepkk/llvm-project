//===- AffineLoopTransform.cpp - Test dep analysis ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to run pair-wise memref access dependence checks.
//
//===----------------------------------------------------------------------===//

#include "iostream"
#include "math.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-loop-transform"

using namespace mlir;

namespace {

unsigned computeRank(std::vector<std::vector<double>> accessMatrix) {
  const double EPS = 1E-9;
  int n = accessMatrix.size();
  int m = accessMatrix[0].size();

  int rank = 0;
  std::vector<bool> row_selected(n, false);
  for (int i = 0; i < m; ++i) {
    int j;
    for (j = 0; j < n; ++j) {
      if (!row_selected[j] && abs(accessMatrix[j][i]) > EPS)
        break;
    }

    if (j != n) {
      ++rank;
      row_selected[j] = true;
      for (int p = i + 1; p < m; ++p)
        accessMatrix[j][p] /= accessMatrix[j][i];
      for (int k = 0; k < n; ++k) {
        if (k != j && abs(accessMatrix[k][i]) > EPS) {
          for (int p = i + 1; p < m; ++p)
            accessMatrix[k][p] -= accessMatrix[j][p] * accessMatrix[k][i];
        }
      }
    }
  }
  return rank;
}

double computeTemporalScore(std::vector<SmallVector<int64_t, 8>> accessMatrix) {
  // TODO: currently not handeling cases such as a[i+j][i+j]
  // Go across rows and check for the number of cols which are all zeros.
  unsigned rows = accessMatrix.size();
  unsigned cols = accessMatrix[0].size();
  int64_t isZero;
  double score = 0.0;
  for (unsigned i = 0; i < cols; ++i) {
    isZero = 0;
    for (unsigned j = 0; j < rows; ++j) {
      // Check is logic slips if some (-)ve value is encountered.
      isZero |= accessMatrix[j][i];
    }
    if (!isZero) {
      // Compute the temporal re-use score as the sum of (index of zero col +
      // 1).
      score += (double)(i + 1);
    }
  }
  return score;
}

double computeSpatialScore(std::vector<SmallVector<int64_t, 8>> accessMatrix,
                           std::vector<double> permuteMap) {
  // Go across rows and check for the number of cols which are all zeros, except
  // the last row.
  unsigned rows = accessMatrix.size();
  unsigned cols = accessMatrix[0].size();
  int64_t isZero;
  double score = 0.0;
  for (unsigned i = 0; i < cols; ++i) {
    isZero = 0;
    for (unsigned j = 0; j < rows - 1; ++j) {
      isZero |= accessMatrix[j][i];
    }
    if (!isZero && accessMatrix[rows - 1][i] != 0) {
      score += (double)permuteMap[i] / abs((double)accessMatrix[rows - 1][i]);
    }
  }
  return score;
}

struct AffineLoopTransform
    : public PassWrapper<AffineLoopTransform, FunctionPass> {
  typedef struct {
    // Encodes if the loop is rectangular or not.
    bool isRectangular = true;
    // Encodes info of all loads and stores.
    typedef struct {
      SmallVector<Operation *, 4> loadsAndStores;
      std::vector<std::vector<SmallVector<int64_t, 8>>> B;
      std::vector<std::vector<SmallVector<int64_t, 8>>> b;
      std::vector<SmallVector<Operation *, 4>> refGroups;
      std::vector<unsigned> ranks;
      std::vector<double> temporalScores;
      typedef struct {
        SmallVector<DependenceComponent, 2> dependenceComponents;
        bool isStatic;
      } Dependence;
      typedef struct {
      public:
        mlir::Operation *srcOpInst;
        mlir::Operation *dstOpInst;
        SmallVector<int64_t, 4> dependence;
      } DependencePairs;
      // Captures all the RAR dependeces along with there dist Vecotrs.
      std::vector<DependencePairs> rarDependences;
      // Captures all the WAR, WAW, RAW dependeces along with there dist
      // Vecotrs.
      std::vector<DependencePairs> wrwDependences;
      // Captures all the dependences in the loopnest.
      std::vector<Dependence> dependences;
      // Captures all the dependences in the form of a dependence matrix.
      SmallVector<SmallVector<int64_t, 4>, 4> dependenceMatrix;
      // Captures all valid interchange Permutations.
      std::vector<std::vector<int64_t>> validPermuataions;
      llvm::DenseMap<int, int> permScore;
      // Utility to insert into dependeces.
      void insertIntoDependences(
          SmallVector<DependenceComponent, 2> dependenceComponents,
          bool isStatic) {
        Dependence toInsert;
        toInsert.dependenceComponents = dependenceComponents;
        toInsert.isStatic = isStatic;
        dependences.push_back(toInsert);
      }
      // Utility to insert into rarDependences.
      void insertIntoRarDependences(mlir::Operation *srcOpInst,
                                    mlir::Operation *dstOpInst,
                                    SmallVector<int64_t, 4> depVector) {
        DependencePairs toInsert;
        toInsert.srcOpInst = srcOpInst;
        toInsert.dstOpInst = dstOpInst;
        toInsert.dependence = depVector;
        rarDependences.push_back(toInsert);
      }
      // Utility to insert into rarDependences.
      void insertIntoWrwDependences(mlir::Operation *srcOpInst,
                                    mlir::Operation *dstOpInst,
                                    SmallVector<int64_t, 4> depVector) {
        DependencePairs toInsert;
        toInsert.srcOpInst = srcOpInst;
        toInsert.dstOpInst = dstOpInst;
        toInsert.dependence = depVector;
        wrwDependences.push_back(toInsert);
      }
    } LoadStoreInfo;
    LoadStoreInfo loadStoreInfo;
    // Encodes info of all loops ina loop nest.
    SmallVector<AffineForOp, 4> loops;
  } LoopInfo;
  // Captures all perfect loops present in the IR.
  std::vector<LoopInfo> perfectLoopNests;
  std::vector<LoopInfo> imperfectLoopNests;
  void runOnFunction() override;
};

// Computes the iteration domain for 'opInst' and populates 'indexSet', which
// encapsulates the constraints involving loops surrounding 'opInst' and
// potentially involving any Function symbols. The dimensional identifiers in
// 'indexSet' correspond to the loops surrounding 'op' from outermost to
// innermost.
// TODO(andydavis) Add support to handle IfInsts surrounding 'op'.
static LogicalResult getInstIndexSet(Operation *op,
                                     FlatAffineConstraints *indexSet) {
  // TODO(andydavis) Extend this to gather enclosing IfInsts and consider
  // factoring it out into a utility function.
  SmallVector<AffineForOp, 4> loops;
  getLoopIVs(*op, &loops);
  return getIndexSet(loops, indexSet);
}

// ValuePositionMap manages the mapping from Values which represent dimension
// and symbol identifiers from 'src' and 'dst' access functions to positions
// in new space where some Values are kept separate (using addSrc/DstValue)
// and some Values are merged (addSymbolValue).
// Position lookups return the absolute position in the new space which
// has the following format:
//
//   [src-dim-identifiers] [dst-dim-identifiers] [symbol-identifiers]
//
// Note: access function non-IV dimension identifiers (that have 'dimension'
// positions in the access function position space) are assigned as symbols
// in the output position space. Convenience access functions which lookup
// an Value in multiple maps are provided (i.e. getSrcDimOrSymPos) to handle
// the common case of resolving positions for all access function operands.
//
// TODO(andydavis) Generalize this: could take a template parameter for
// the number of maps (3 in the current case), and lookups could take indices
// of maps to check. So getSrcDimOrSymPos would be "getPos(value, {0, 2})".
class ValuePositionMap1 {
public:
  void addSrcValue(Value value) {
    if (addValueAt(value, &srcDimPosMap, numSrcDims))
      ++numSrcDims;
  }
  void addSymbolValue(Value value) {
    if (addValueAt(value, &symbolPosMap, numSymbols))
      ++numSymbols;
  }
  unsigned getSrcDimOrSymPos(Value value) const {
    return getDimOrSymPos(value, srcDimPosMap, 0);
  }
  unsigned getSymPos(Value value) const {
    auto it = symbolPosMap.find(value);
    assert(it != symbolPosMap.end());
    return numSrcDims + it->second;
  }

  unsigned getNumSrcDims() const { return numSrcDims; }
  unsigned getNumDims() const { return numSrcDims; }
  unsigned getNumSymbols() const { return numSymbols; }

private:
  bool addValueAt(Value value, DenseMap<Value, unsigned> *posMap,
                  unsigned position) {
    auto it = posMap->find(value);
    if (it == posMap->end()) {
      (*posMap)[value] = position;
      return true;
    }
    return false;
  }
  unsigned getDimOrSymPos(Value value,
                          const DenseMap<Value, unsigned> &dimPosMap,
                          unsigned dimPosOffset) const {
    auto it = dimPosMap.find(value);
    if (it != dimPosMap.end()) {
      return dimPosOffset + it->second;
    }
    it = symbolPosMap.find(value);
    assert(it != symbolPosMap.end());
    return numSrcDims + it->second;
  }

  unsigned numSrcDims = 0;
  unsigned numSymbols = 0;
  DenseMap<Value, unsigned> srcDimPosMap;
  DenseMap<Value, unsigned> symbolPosMap;
};
// Builds a map from Value to identifier position in a new merged identifier
// list, which is the result of merging dim/symbol lists from src/dst
// iteration domains, the format of which is as follows:
//
//   [src-dim-identifiers, dst-dim-identifiers, symbol-identifiers, const_term]
//
// This method populates 'valuePosMap' with mappings from operand Values in
// 'srcAccessMap'/'dstAccessMap' (as well as those in 'srcDomain'/'dstDomain')
// to the position of these values in the merged list.
static void
buildDimAndSymbolPositionMaps1(const FlatAffineConstraints &srcDomain,
                               const AffineValueMap &srcAccessMap,
                               ValuePositionMap1 *valuePosMap,
                               FlatAffineConstraints *dependenceConstraints) {
  auto updateValuePosMap = [&](ArrayRef<Value> values, bool isSrc) {
    for (unsigned i = 0, e = values.size(); i < e; ++i) {
      auto value = values[i];
      if (!isForInductionVar(values[i])) {
        assert(isValidSymbol(values[i]) &&
               "access operand has to be either a loop IV or a symbol");
        valuePosMap->addSymbolValue(value);
      } else if (isSrc) {
        valuePosMap->addSrcValue(value);
      } else {
        /*nothing*/
      }
    }
  };

  SmallVector<Value, 4> srcValues;
  srcDomain.getIdValues(0, srcDomain.getNumDimAndSymbolIds(), &srcValues);
  // Update value position map with identifiers from src iteration domain.
  updateValuePosMap(srcValues, /*isSrc=*/true);
  // Update value position map with identifiers from src access function.
  updateValuePosMap(srcAccessMap.getOperands(), /*isSrc=*/true);
}

// Sets up dependence constraints columns appropriately, in the format:
// [src-dim-ids, dst-dim-ids, symbol-ids, local-ids, const_term]
static void
initDependenceConstraints1(const FlatAffineConstraints &srcDomain,
                           const AffineValueMap &srcAccessMap,
                           const ValuePositionMap1 &valuePosMap,
                           FlatAffineConstraints *dependenceConstraints) {
  // Calculate number of equalities/inequalities and columns required to
  // initialize FlatAffineConstraints for 'dependenceDomain'.
  unsigned numIneq =
      srcDomain.getNumInequalities(); // + dstDomain.getNumInequalities();
  AffineMap srcMap = srcAccessMap.getAffineMap();
  // assert(srcMap.getNumResults() ==
  // dstAccessMap.getAffineMap().getNumResults());
  unsigned numEq = srcMap.getNumResults();
  unsigned numDims = srcDomain.getNumDimIds();
  unsigned numSymbols = 0; // valuePosMap.getNumSymbols();
  unsigned numLocals = 0;  // srcDomain.getNumLocalIds();
  unsigned numIds = numDims + numSymbols + numLocals;
  unsigned numCols = numIds + 1;

  // Set flat affine constraints sizes and reserving space for constraints.
  dependenceConstraints->reset(numIneq, numEq, numCols, numDims, numSymbols,
                               numLocals);

  // Set values corresponding to dependence constraint identifiers.
  SmallVector<Value, 4> srcLoopIVs;
  srcDomain.getIdValues(0, srcDomain.getNumDimIds(), &srcLoopIVs);

  dependenceConstraints->setIdValues(0, srcLoopIVs.size(), srcLoopIVs);
  // dependenceConstraints->setIdValues(srcLoopIVs.size(), srcLoopIVs.size() +
  // dstLoopIVs.size(), dstLoopIVs);

  // Set values for the symbolic identifier dimensions.
  auto setSymbolIds = [&](ArrayRef<Value> values) {
    for (auto value : values) {
      if (!isForInductionVar(value)) {
        assert(isValidSymbol(value) && "expected symbol");
        dependenceConstraints->setIdValue(valuePosMap.getSymPos(value), value);
      }
    }
  };

  setSymbolIds(srcAccessMap.getOperands());

  SmallVector<Value, 8> srcSymbolValues;
  srcDomain.getIdValues(srcDomain.getNumDimIds(),
                        srcDomain.getNumDimAndSymbolIds(), &srcSymbolValues);
  setSymbolIds(srcSymbolValues);

  for (unsigned i = 0, e = dependenceConstraints->getNumDimAndSymbolIds();
       i < e; i++)
    assert(dependenceConstraints->getIds()[i].hasValue());
}

static LogicalResult
addMemRefAccessConstraints1(const AffineValueMap &srcAccessMap,
                            const ValuePositionMap1 &valuePosMap,
                            FlatAffineConstraints *dependenceDomain,
                            AffineLoopTransform::LoopInfo *loopInfo) {
  AffineMap srcMap = srcAccessMap.getAffineMap();
  // assert(srcMap.getNumResults() == dstMap.getNumResults());
  unsigned numResults = srcMap.getNumResults();
  // print number of results.
  // std::cout<<"number of results in srcMap: "<<srcMap.getNumResults()<<"\n";
  // std::cout<<"number of dims in srcMap: "<<srcMap.getNumLocalIds()<<"\n";
  unsigned srcNumIds = srcMap.getNumDims() + srcMap.getNumSymbols();
  ArrayRef<Value> srcOperands = srcAccessMap.getOperands();

  std::vector<SmallVector<int64_t, 8>> srcFlatExprs;
  FlatAffineConstraints srcLocalVarCst;
  // Get flattened expressions for the source destination maps.
  if (failed(getFlattenedAffineExprs(srcMap, &srcFlatExprs, &srcLocalVarCst)))
    return failure();
  // Prints a Sort of accessFunction for the access map.
  /*
std::cout << "printing srcFlatExprs" << std::endl;
for (auto i : srcFlatExprs) {
for (auto j : i) {
std::cout << j << " ";
}
std::cout << std::endl;
}
  // Prints nothing for some reason.
std::cout << "printing srcLocalVarCst" << std::endl;
  srcLocalVarCst.dump();
  */
  unsigned domNumLocalIds = dependenceDomain->getNumLocalIds();
  unsigned srcNumLocalIds = srcLocalVarCst.getNumLocalIds();
  unsigned numLocalIdsToAdd = srcNumLocalIds;
  for (unsigned i = 0; i < numLocalIdsToAdd; i++) {
    dependenceDomain->addLocalId(dependenceDomain->getNumLocalIds());
  }
  unsigned numDims = dependenceDomain->getNumDimIds();
  unsigned numSymbols = dependenceDomain->getNumSymbolIds();
  // Commented by myself.
  // unsigned numSrcLocalIds = srcLocalVarCst.getNumLocalIds();
  unsigned newLocalIdOffset = numDims + numSymbols + domNumLocalIds;

  // Equality to add.
  SmallVector<int64_t, 8> eq(dependenceDomain->getNumCols());
  // 2 composite vectors B and b which will have the access function.
  std::vector<SmallVector<int64_t, 8>> B;
  std::vector<SmallVector<int64_t, 8>> b;
  for (unsigned i = 0; i < numResults; ++i) {
    // Zero fill.
    std::fill(eq.begin(), eq.end(), 0);

    // Flattened AffineExpr for src result 'i'.
    const auto &srcFlatExpr = srcFlatExprs[i];
    // Set identifier coefficients from src access function.
    for (unsigned j = 0, e = srcOperands.size(); j < e; ++j)
      eq[valuePosMap.getSrcDimOrSymPos(srcOperands[j])] = srcFlatExpr[j];
    // std::cout << "eq size first loop: " << eq.size() << "\n";
    // Local terms.
    for (unsigned j = 0, e = srcNumLocalIds; j < e; j++)
      eq[newLocalIdOffset + j] = srcFlatExpr[srcNumIds + j];
    // std::cout << "eq size second loop: " << eq.size() << "\n";
    // Insert the equation into the loopNest.LoadstoreInfo.B
    // loopInfo->loadStoreInfo.B.push_back(eq);
    // Set constant term.
    eq[eq.size() - 1] = srcFlatExpr[srcFlatExpr.size() - 1];

    // Insert the equation into the loopNest.LoadstoreInfo.B
    B.push_back(SmallVector<int64_t, 8>(
        eq.begin(),
        eq.end() - numSymbols - dependenceDomain->getNumLocalIds() - 1));
    // Insert the equation into the loopNest.LoadstoreInfo.b
    b.push_back(SmallVector<int64_t, 8>(eq.end() - 1, eq.end()));

    // Add equality constraint.
    dependenceDomain->addEquality(eq);
  }
  loopInfo->loadStoreInfo.B.push_back(B);
  loopInfo->loadStoreInfo.b.push_back(b);
  B.clear();
  b.clear();

  // Add equality constraints for any operands that are defined by constant ops.
  auto addEqForConstOperands = [&](ArrayRef<Value> operands) {
    for (unsigned i = 0, e = operands.size(); i < e; ++i) {
      if (isForInductionVar(operands[i]))
        continue;
      auto symbol = operands[i];
      assert(isValidSymbol(symbol));
      // Check if the symbol is a constant.
      if (auto cOp = dyn_cast_or_null<ConstantIndexOp>(symbol.getDefiningOp()))
        dependenceDomain->setIdToConstant(valuePosMap.getSymPos(symbol),
                                          cOp.getValue());
    }
  };

  // Add equality constraints for any src symbols defined by constant ops.
  addEqForConstOperands(srcOperands);

  // By construction (see flattener), local var constraints will not have any
  // equalities.
  assert(srcLocalVarCst.getNumEqualities() == 0);

  // Add inequalities from srcLocalVarCst and destLocalVarCst into the
  // dependence domain.
  /*
  SmallVector<int64_t, 8> ineq(dependenceDomain->getNumCols());
  for (unsigned r = 0, e = srcLocalVarCst.getNumInequalities(); r < e; r++) {
    std::fill(ineq.begin(), ineq.end(), 0);

    // Set identifier coefficients from src local var constraints.
    for (unsigned j = 0, e = srcOperands.size(); j < e; ++j)
      ineq[valuePosMap.getSrcDimOrSymPos(srcOperands[j])] =
          srcLocalVarCst.atIneq(r, j);
    // Local terms.
    for (unsigned j = 0, e = srcNumLocalIds; j < e; j++)
      ineq[newLocalIdOffset + j] = srcLocalVarCst.atIneq(r, srcNumIds + j);
    // Set constant term.
    ineq[ineq.size() - 1] =
        srcLocalVarCst.atIneq(r, srcLocalVarCst.getNumCols() - 1);
    dependenceDomain->addInequality(ineq);
  }
*/
  return success();
}

} // end anonymous namespace

// Finds all only the perfect loop nests inside a function if mode is set to
// false, if mode is set to true finds imperfect loop nests.

static void getPerfectOrImperfectLoopNests(
    SmallVectorImpl<AffineForOp> &forOps, AffineForOp rootForOp,
    unsigned maxLoops = std::numeric_limits<unsigned>::max()) {
  for (unsigned i = 0; i < maxLoops; ++i) {
    forOps.push_back(rootForOp);
    Block &body = rootForOp.region().front();
    // NOT SURE, but seems to be checking if the loop is first and only non
    // terminator. There will be two things in the block, some Opeartion(an
    // AffineForOp)
    // and a terminator operation. Hence a (-2) in std::prev.
    if (body.begin() != std::prev(body.end(), 2)) {
		 /*	
			std::cout<<"first condition fail: \n";
			// Trying to make a copy of this for op.
			mlir::OpBuilder builder(rootForOp.getParentOfType<FuncOp>().getBody());
			auto lbMap = rootForOp.getLowerBoundMap();
			SmallVector<Value, 4> lbOperands(rootForOp.getLowerBoundOperands());
			augmentMapAndBounds(builder, rootForOp.getInductionVar(), &lbMap, &lbOperands);

			auto forOp = builder.create<AffineForOp>();
      // I still want to continue the search. I want to set the rootForOp to the
      // next AffineForOp if any. Iterate this block and try to find an
      // AffinForOp.
      for (auto op = body.begin(); op != body.end(); op++) {
        if (isa<AffineForOp>(*op)) {
					std::cout<<"for Loop found: \n";
          rootForOp = dyn_cast<AffineForOp>(*op);
          break;
        }
      }
			if(rootForOp)
				continue;
			*/
      // return;
    }
    // Checking if the first operation in body is an AffineForOP.
    // If 'yes' continue, else if searching for perfect loop nests,
    // stop search. if the nest operation is not an affine for op then 
    // body loop is imperfeclty nested.
    rootForOp = dyn_cast<AffineForOp>(&body.front());
    if (!rootForOp) {
		//	std::cout<<"second condition fail: \n";
      // I still want to continue the search. I want to set the rootForOp to the
      // next AffineForOp if any. Iterate this block and try to find an
      // AffinForOp.
      for (auto op = body.begin(); op != body.end(); op++) {
        if (isa<AffineForOp>(*op)) {
          rootForOp = dyn_cast<AffineForOp>(*op);
          break;
        }
      }
      // If rootForOp is still NULL no AffineForOp was present hence return.
      if (!rootForOp) {
        return;
      }
    }
  }
}

static void
getLoopNests(FuncOp f, std::vector<AffineLoopTransform::LoopInfo> *loopNests) {
  auto getImPerfectLoopNests = [&](AffineForOp root) {
    AffineLoopTransform::LoopInfo capturedLoops;
    getPerfectOrImperfectLoopNests(capturedLoops.loops, root);
    loopNests->push_back(capturedLoops);
  };
  for (auto &block : f)
    for (auto &op : block)
      if (auto forOp = dyn_cast<AffineForOp>(op))
        getImPerfectLoopNests(forOp);
}

// Returns a result string which represents the direction vector (if there was
// a dependence), returns the string "false" otherwise.
static std::string
getDirectionVectorStr(bool ret, unsigned numCommonLoops, unsigned loopNestDepth,
                      ArrayRef<DependenceComponent> dependenceComponents) {
  if (!ret)
    return "false";
  if (dependenceComponents.empty() || loopNestDepth > numCommonLoops)
    return "true";
  std::string result;
  for (unsigned i = 0, e = dependenceComponents.size(); i < e; ++i) {
    std::string lbStr = "-inf";
    if (dependenceComponents[i].lb.hasValue() &&
        dependenceComponents[i].lb.getValue() !=
            std::numeric_limits<int64_t>::min()) {
      lbStr = std::to_string(dependenceComponents[i].lb.getValue());
    }

    std::string ubStr = "+inf";
    if (dependenceComponents[i].ub.hasValue() &&
        dependenceComponents[i].ub.getValue() !=
            std::numeric_limits<int64_t>::max()) {
      ubStr = std::to_string(dependenceComponents[i].ub.getValue());
    }
    // DependenceComponent contains state about the direction of a dependence as
    // an interval [lb, ub] for an AffineForOp. Distance vectors components are
    // represented by the interval [lb, ub] with lb == ub. Direction vectors
    // components are represented by the interval [lb, ub] with lb < ub. Note
    // that ub/lb == None means unbounded.
    result += "[" + lbStr + ", " + ubStr + "]";
  }
  return result;
}

static void
condenseDependences(SmallVector<DependenceComponent, 2> dependenceComponents,
                    SmallVector<int64_t, 4> *depVector) {
  for (unsigned i = 0, e = dependenceComponents.size(); i < e; ++i) {
    int64_t lb = std::numeric_limits<int64_t>::min();
    int64_t ub = lb;
    if (dependenceComponents[i].lb.hasValue() &&
        dependenceComponents[i].lb.getValue() !=
            std::numeric_limits<int64_t>::min()) {
      lb = dependenceComponents[i].lb.getValue();
    }
    if (dependenceComponents[i].lb.hasValue() &&
        dependenceComponents[i].lb.getValue() !=
            std::numeric_limits<int64_t>::min()) {
      ub = dependenceComponents[i].ub.getValue();
    }
    if (lb != std::numeric_limits<int64_t>::min() &&
        ub != std::numeric_limits<int64_t>::min()) {
      if (lb == ub)
        depVector->push_back(lb);
      else if (lb < 0 && ub < 0)
        depVector->push_back(ub);
      else if (lb > 0 && ub > 0)
        depVector->push_back(lb);
      else if (lb == 0 && ub > 0)
        depVector->push_back(1);
      else if (lb < 0 && ub == 0)
        depVector->push_back(-1);
      else
        // TODO: check this if test case 2 fails.
        depVector->push_back(-1);
    }
  }
}

// For each access in 'loadsAndStores', runs a dependence check between this
// "source" access and all subsequent "destination" accesses in
// 'loadsAndStores'. Emits the result of the dependence check as a note with
// the source access.
// Modifying to my needs. Passing in the reference of the loop nest which
// contains the loads/stroes so that it is easy to push the dependences.
static void checkDependences(AffineLoopTransform::LoopInfo *loopNest) {
  SmallVector<Operation *, 4> loadsAndStores =
      loopNest->loadStoreInfo.loadsAndStores;
  for (unsigned i = 0, e = loadsAndStores.size(); i < e; ++i) {
    auto *srcOpInst = loadsAndStores[i];
    MemRefAccess srcAccess(srcOpInst);
    for (unsigned j = 0; j < e; ++j) {
      auto *dstOpInst = loadsAndStores[j];
      MemRefAccess dstAccess(dstOpInst);

      unsigned numCommonLoops =
          getNumCommonSurroundingLoops(*srcOpInst, *dstOpInst);
      for (unsigned d = 1; d <= numCommonLoops + 1; ++d) {
        FlatAffineConstraints dependenceConstraints;
        SmallVector<DependenceComponent, 2> dependenceComponents;
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, d, &dependenceConstraints,
            &dependenceComponents, true);
        assert(result.value != DependenceResult::Failure);
        bool ret = hasDependence(result);
        std::string isDependence =
            getDirectionVectorStr(ret, numCommonLoops, d, dependenceComponents);

        // Can extract the access functions here from the returned
        // FlatAffineCostraints, irrespective of whether dependence is present
        // or not. The equalitites in the structure will eventually represent
        // the access functions. Let's print and see them.

        if (isDependence.compare("false") == 0) {
          // Dependence does not exists.
        } else if (isDependence.compare("true") == 0) {
          // Static dependence exists.
          // srcOpInst->emitRemark(" Static dependence from ")
          //    << i << " to " << j << " at depth " << d << " = "
          //    << getDirectionVectorStr(ret, numCommonLoops, d,
          //                             dependenceComponents);
          loopNest->loadStoreInfo.insertIntoDependences(dependenceComponents,
                                                        /*isStatic*/ true);
          SmallVector<int64_t, 4> depVector;
          condenseDependences(dependenceComponents, &depVector);
          // Handle RAR dependences differently.
          // Don't push RAR into the dependence vector.
          if (isa<AffineLoadOp>(srcOpInst) && isa<AffineLoadOp>(dstOpInst)) {
            loopNest->loadStoreInfo.insertIntoRarDependences(
                srcOpInst, dstOpInst, depVector);
          } else {
            // Once the dependence is found insert it into a rarDependences
            // vector.
            //std::cout << "static dependence: \n";
            //srcOpInst->dump();
            //dstOpInst->dump();
            //std::cout << "static dependence: \n";
            loopNest->loadStoreInfo.insertIntoWrwDependences(
                srcOpInst, dstOpInst, depVector);
          }
          // Compute the dependence vectors as needed.
          getDirectionVectorStr(ret, numCommonLoops, d, dependenceComponents);
          // Find no more dependences at further depths BREAK here.
          // break;
        } else {
          // Dynamic dependence exists.
          // srcOpInst->emitRemark(" Dynamic dependence from ")
          //    << i << " to " << j << " at depth " << d << " = "
          //    << getDirectionVectorStr(ret, numCommonLoops, d,
          //                             dependenceComponents);
          loopNest->loadStoreInfo.insertIntoDependences(dependenceComponents,
                                                        /*isStatic*/ false);
          // Compute the dependence vectors as needed.
          SmallVector<int64_t, 4> depVector;
          condenseDependences(dependenceComponents, &depVector);
          // Handle RAR dependences differently.
          // Don't push RAR into the dependence vector.
          if (isa<AffineLoadOp>(srcOpInst) && isa<AffineLoadOp>(dstOpInst)) {
            // std::cout<<"RAR found between: \n";
            // srcOpInst->dump();
            // dstOpInst->dump();
            // for(auto x : depVector)
            //	std::cout<<x<<" ";
            // std::cout<<"\n";
            // Once the dependence is found insert it into a rarDependences
            // vector.
            loopNest->loadStoreInfo.insertIntoRarDependences(
                srcOpInst, dstOpInst, depVector);
          } else {
            loopNest->loadStoreInfo.dependenceMatrix.push_back(depVector);
            // Once the dependence is found insert it into a rarDependences
            // vector.
            loopNest->loadStoreInfo.insertIntoWrwDependences(
                srcOpInst, dstOpInst, depVector);
          }
          // Find no more dependences at further depths BREAK here.
          // break;
        }

        // TODO(andydavis) Print dependence type (i.e. RAW, etc) and print
        // distance vectors as: ([2, 3], [0, 10]). Also, shorten distance
        // vectors from ([1, 1], [3, 3]) to (1, 3).
        // srcOpInst->emitRemark("dependence from ")
        //    << i << " to " << j << " at depth " << d << " = "
        //    << getDirectionVectorStr(ret, numCommonLoops, d,
        //                             dependenceComponents);
      }
    }
  }
}

static void
eliminateInvalidDependence(AffineLoopTransform::LoopInfo *loopInfo,
                           std::vector<std::vector<int64_t>> allPermutaions) {
  bool isValid;
  SmallVector<SmallVector<int64_t, 4>, 4> dependenceMatrix =
      loopInfo->loadStoreInfo.dependenceMatrix;
  for (auto perm : allPermutaions) {
    // Check is the selected permutaion is invalid by checking if any
    // depenedence in dependenceMatrix becomes lexicographically less than zero.
    isValid = true;
    for (unsigned i = 0; i < dependenceMatrix.size(); i++) {
      for (unsigned j = 0; j < dependenceMatrix[0].size(); j++) {
        if (dependenceMatrix[i][perm[j]] == 0)
          continue;
        else if (dependenceMatrix[i][perm[j]] > 0)
          break;
        else {
          isValid = false;
          break;
        }
      }
      if (!isValid)
        break;
    }
    if (isValid) {
      loopInfo->loadStoreInfo.validPermuataions.push_back(perm);
    }
  }
}
static bool
compareAccessMatrix(std::vector<SmallVector<int64_t, 8>> srcMatrix,
                    std::vector<SmallVector<int64_t, 8>> dstMatrix) {
  bool isEqual = true;
  // First check if the  dimensions of srcMatrix and dstMatrix are different,
  // If they are different then false can be returned.
  if ((srcMatrix.size() != dstMatrix.size()) ||
      (srcMatrix[0].size() != dstMatrix[0].size())) {
    return false;
  }
  for (unsigned i = 0; i < srcMatrix.size(); i++) {
    for (unsigned j = 0; j < srcMatrix[0].size(); j++) {
      if (srcMatrix[i][j] != dstMatrix[i][j]) {
        isEqual = false;
      }
    }
  }
  return isEqual;
}

static void createRefGroups(AffineLoopTransform::LoopInfo *loopNest,
                            std::vector<int64_t> permuteMap) {
	// At one time only one refGroup can be present.
	// Clear refgroup before starting.
	loopNest->loadStoreInfo.refGroups.clear();
  // First create treat all the individual access as one reference group.
  // Initialize a boolvector 'isVisited' to mark all the loads/stores as not
  // visited.
  SmallVector<mlir::Operation *, 4> toPush;
  std::vector<bool> isVisited;
  for (auto &loadOrStore : loopNest->loadStoreInfo.loadsAndStores) {
    toPush.push_back(loadOrStore);
    loopNest->loadStoreInfo.refGroups.push_back(toPush);
    toPush.clear();
    isVisited.push_back(false);
  }
  //std::cout << "size before group creation: "
         //   << loopNest->loadStoreInfo.refGroups.size() << std::endl;

  // Now refGroups has been intialized to contain  all accesses as individual
  // groups.
  while (1) {
    int toContinue = -1;
    // Check if something is unvisited.
    for (unsigned i = 0; i < isVisited.size(); ++i) {
      if (!isVisited[i]) {
        toContinue = (int)i;
        break;
      }
    }
    // Nothing is unvisited break.
    if (toContinue == -1)
      break;
    else {
      // std::cout<<"To continue: "<<toContinue<<"\n";
      isVisited[toContinue] = true;
			// Find the size of cache line size.
			auto loadOp = dyn_cast<AffineLoadOp>(loopNest->loadStoreInfo.loadsAndStores[toContinue]); 
			auto storeOp = dyn_cast<AffineStoreOp>(loopNest->loadStoreInfo.loadsAndStores[toContinue]);
			unsigned width;
			if(loadOp){
				width = loadOp.getMemRefType().getElementType().getIntOrFloatBitWidth(); 
			}
			else{
				width = storeOp.getMemRefType().getElementType().getIntOrFloatBitWidth(); 
			}
			unsigned cls = 64 / (width / 8);
      // Now we have the index of the operation. I can go ahead and see if this
      // opertion has same access function as that of some other group, except
      // itself.
      for (auto &refGroup : loopNest->loadStoreInfo.refGroups) {
        // checking here if the refGroup are same as the chosen ref. If it is
        // the same then skip this iteration.
        bool isSameRefGroup = false;
        for (auto &ref : refGroup) {
          if (ref == loopNest->loadStoreInfo.loadsAndStores[toContinue])
            isSameRefGroup = true;
        }
        // If refGroup is same then continue.
        if (isSameRefGroup)
          continue;
        // TODO currently choosing first access as the representative of the
        // refGroup. Should have chosen the worst access.
        // Find the index of therepresentative of the reference group in the
        // loadsOrStore list.
        unsigned repInx;
        for (repInx = 0; repInx < loopNest->loadStoreInfo.loadsAndStores.size();
             repInx++) {
          if (refGroup[0] == loopNest->loadStoreInfo.loadsAndStores[repInx]) {
            break;
          }
        }
        bool toGroup = false;
        // Compare the access matrix, of the current candidate to be grouped and
        // the refGroup representative. If true is returned then the accesses
        // can potentially be grouped by checking further conditions.
        if (compareAccessMatrix(loopNest->loadStoreInfo.B[repInx],
                                loopNest->loadStoreInfo.B[toContinue])) {
          // Check if these accesses are present in any RAR, RAW, WAR, WAW
          // dependencies. First checking in the rarDependences. have to chekc
          // bothways. by first keeping one access as source and ther as
          // destination and viceversa.
          mlir::Operation *srcOpInst =
              loopNest->loadStoreInfo.loadsAndStores[repInx];
          mlir::Operation *dstOpInst =
              loopNest->loadStoreInfo.loadsAndStores[toContinue];
          // First checking in RAR.
          for (auto &dep : loopNest->loadStoreInfo.rarDependences) {
            // Solution exists for if true. We can then check if spatial or
            // temporal is present.
            if (((dep.srcOpInst == srcOpInst && dep.dstOpInst == dstOpInst) ||
                 (dep.srcOpInst == dstOpInst && dep.dstOpInst == srcOpInst)) &&
                (srcOpInst != dstOpInst)) {
              //std::cout << "RAR dep found: \n";
              //srcOpInst->getLoc().dump();
              //dstOpInst->getLoc().dump();
              //for (auto dep : dep.dependence) {
              //  std::cout << dep << " ";
              //}
              //std::cout << "\n";
              // Check for temproal resue. if only varies in the last dimension
              // of the dependence only then we can say that temporal reuse is
              // present.
              bool isZero = true;
              for (unsigned i = 0; i < dep.dependence.size() - 1; ++i) {
                if (dep.dependence[permuteMap[i]] != 0)
                  isZero = false;
              }
              //std::cout << "isZero after temporal check: " << isZero << "\n";
              // if all the elements are 0 uptil the last element then temporal
              // reuse is present.
              if (isZero) {
                // TODO: add check for the condition given in paper for temporal
                // reuse.
                // The condition essentially means that temporal reuse can even
                // be exploited if there is agap of atmost '2' between
                // iterations.
                if (abs(dep.dependence[permuteMap[dep.dependence.size() -
                                                  1]]) <= 2) {
                  toGroup = true;
                  break;
                }
              }
            }
          }
          // If we have found that the group can be made then break. and dont
          // check for WRW, and then check if theere is no dependence but still
          // there is spatial re-use because of the last dimension just varying
          // a little(less than 'cls').
          if (!toGroup) {
            // then checking in WRW.
            for (auto dep : loopNest->loadStoreInfo.wrwDependences) {
              // Solution exists for if true. We can then check if spatial or
              // temporal is present.
              if (((dep.srcOpInst == srcOpInst && dep.dstOpInst == dstOpInst) ||
                   (dep.srcOpInst == dstOpInst &&
                    dep.dstOpInst == srcOpInst)) &&
                  (srcOpInst != dstOpInst)) {
                //std::cout << "wrw dep found: \n";
                //srcOpInst->getLoc().dump();
                //dstOpInst->getLoc().dump();
                //for (auto dep : dep.dependence) {
                  //std::cout << dep << " ";
                //}
                //std::cout << "\n";
                // Check for temproal resue. if only varies in the last
                // dimension of the dependence only then we can say that
                // temporal reuse is present.
                bool isZero = true;
                for (unsigned i = 0; i < dep.dependence.size() - 1; ++i) {
                  if (dep.dependence[permuteMap[i]] != 0)
                    isZero = false;
                }
                // if all the elements are 0 uptil the last element then
                // temporal reuse is present.
                if (isZero) {
                  if (abs(dep.dependence[permuteMap[dep.dependence.size() -
                                                    1]]) <= 2) {
                    toGroup = true;
                    break;
                  }
                  //std::cout << "toGroup at rar " << toGroup << "\n";
                }
              }
            }
            // Check here for the last condition where the access matrix are the
            // same, but there is no dependence but still we can pair the
            // elements because of spatial reuse being present. to check the
            // spatial reuse we needto check the 'b', if it is the same overall
            // and only the entry corresponding to the last loop varies by a
            // small amount(cls)
            if (!toGroup) {
              //std::cout << "not grouped until now, last try: \n";
              // We already know th access matrix are the same. just check the
              // last value in the matrix and see if the constants just vary by
              // a small constant.
              bool isSame = true;
              std::vector<SmallVector<int64_t, 8>> b1 =
                  loopNest->loadStoreInfo.b[toContinue];
              std::vector<SmallVector<int64_t, 8>> b2 =
                  loopNest->loadStoreInfo.b[repInx];
              for (unsigned i = 0; i < b1.size(); ++i) {
                for (unsigned j = 0; j < b1[i].size() - 1; j++) {
                  if (b1[i][permuteMap[j]] != b2[i][permuteMap[j]])
                    isSame = false;
                }
              }
              if (isSame) {
                // Check if difference in last elements is less than the CLS.
                if (abs(b1[0][permuteMap[b1[0].size() - 1]] -
                        b2[0][permuteMap[b2[0].size() - 1]]) < cls /*CLS*/) {
                  toGroup = true;
                }
              }
            }
          }
        }
        // If to group is set then the elements can be mergerd nto one group
        // erase from the vector adn insert into the refGroup and then break the
        // loop.
        //std::cout << "toGroup at end " << toGroup << "\n";
        if (toGroup) {
          // Find the RefGroup which has the element loadsAndStores[toContinue]
          // and delete it.
          //std::cout << "size before entry deletion: "
          //          << loopNest->loadStoreInfo.refGroups.size() << "\n";
          //loopNest->loadStoreInfo.loadsAndStores[toContinue]->getLoc().dump();
          unsigned refGroupInx;
          bool flag = false;
          for (refGroupInx = 0;
               refGroupInx < loopNest->loadStoreInfo.refGroups.size();
               refGroupInx++) {
            for (auto ls : loopNest->loadStoreInfo.refGroups[refGroupInx]) {
              if (ls == loopNest->loadStoreInfo.loadsAndStores[toContinue]) {
                flag = true;
              }
            }
            if (flag)
              break;
          }
          // Push it into the refGroup.
          refGroup.push_back(
              loopNest->loadStoreInfo.loadsAndStores[toContinue]);
          // Delete it's individual entry from the ref group.
          loopNest->loadStoreInfo.refGroups.erase(
              loopNest->loadStoreInfo.refGroups.begin() + refGroupInx);
          // Mark the thing in RefGroup as visited.
          isVisited[repInx] = true;
          //std::cout << "size after entry deletion: "
          //          << loopNest->loadStoreInfo.refGroups.size() << "\n";
          break;
        }
      }
    }
  }
	std::cout<<"permutatiopn: \n";
	for(auto x: permuteMap)
		std::cout<<x<<" ";
	std::cout<<std::endl;
  std::cout << "size after group creation: "
            << loopNest->loadStoreInfo.refGroups.size() << std::endl;
  for (auto refGroup : loopNest->loadStoreInfo.refGroups) {
    std::cout << "refgroup start:\n";
    for (auto ls : refGroup) {
      ls->getLoc().dump();
    }
    std::cout << "refgroup end:\n";
  }
}

static double computeCacheMisses(AffineLoopTransform::LoopInfo *loopNest, std::vector<int64_t> permuteMap) {
  // Iterate through the representative of the refGroups and start calculating
  // cache miss for each refGroup.
  double totalCost = 0.0f;
  double cost;
  int64_t lb, ub, stride, iter, loopStep;
  for (auto &refGroup : loopNest->loadStoreInfo.refGroups) {
    // Initialize the cost of the Ref Group to be 1.
    cost = 1.0f;
    // TODO: currently taking the first element as the representative of the
    // refGroup. 
    // Find the first element in load/store list.
    unsigned refGroupInx;
    for (refGroupInx = 0;
         refGroupInx < loopNest->loadStoreInfo.loadsAndStores.size();
         ++refGroupInx) {
      if (refGroup[0] == loopNest->loadStoreInfo.loadsAndStores[refGroupInx]) {
        break;
      }
    }
    // To calculate the no of cache misses traverse the access matrix form the
    // last column to the first column.
    std::vector<SmallVector<int64_t, 8>> accessMatrix =
        loopNest->loadStoreInfo.B[refGroupInx];
			// Find the size of cache line size.
			auto loadOp = dyn_cast<AffineLoadOp>(loopNest->loadStoreInfo.loadsAndStores[refGroupInx]); 
			auto storeOp = dyn_cast<AffineStoreOp>(loopNest->loadStoreInfo.loadsAndStores[refGroupInx]);
			unsigned width;
			if(loadOp){
				width = loadOp.getMemRefType().getElementType().getIntOrFloatBitWidth(); 
			}
			else{
				width = storeOp.getMemRefType().getElementType().getIntOrFloatBitWidth(); 
			}
			unsigned cls = 64 / (width / 8);
    // 1-D access matrix needs to be handeled sperately.
    if (accessMatrix.size() == 1) {
			int j;
      for  (j = accessMatrix[0].size() - 1; j >= 0; j--) {
        lb = loopNest->loops[permuteMap[j]].getConstantLowerBound();
        ub = loopNest->loops[permuteMap[j]].getConstantUpperBound();
        loopStep = loopNest->loops[permuteMap[j]].getStep();
        iter = ((ub - 1) - lb + loopStep) / loopStep;
        stride = loopStep * accessMatrix[accessMatrix.size() - 1][permuteMap[j]];
        if (accessMatrix[0][permuteMap[j]] == 0) {
					std::cout<<"last element is zero: \n";
          // If last element is zero then cost is 1.
          cost *= 1;
        } else {
          // Last element is not zero hence some cache misses will be
          // encountered. If the 'stride' is less than 'cls' then some spatial
          // re-use is present.
          if (stride < cls) {
						std::cout<<"last element is lt cls: \n";
            cost *= ((iter / cls) / (stride));
						// break here because all the iterations after point will have n-misses.
						break;
          }
          // If not then no spatial reuse is present and all will be misses.
          else {
						std::cout<<"last element is gt cls: \n";
            cost *= iter;
						// break here because all the iterations after point will have n-misses.
						break;
          }
        }
      }
			for (int l = j - 1; l >= 0; l--) {
				lb = loopNest->loops[permuteMap[l]].getConstantLowerBound();
				ub = loopNest->loops[permuteMap[l]].getConstantUpperBound();
				loopStep = loopNest->loops[permuteMap[l]].getStep();
				iter = ((ub - 1) - lb + loopStep) / loopStep;
				cost *= iter;
			}
      loopNest->loadStoreInfo.loadsAndStores[refGroupInx]->getLoc().dump();
      std::cout << "access matrix cost: " << cost << "\n";
			totalCost += cost;
      continue;
    }
    for (int j = accessMatrix[0].size() - 1; j >= 0; j--) {
      lb = loopNest->loops[permuteMap[j]].getConstantLowerBound();
      ub = loopNest->loops[permuteMap[j]].getConstantUpperBound();
      loopStep = loopNest->loops[permuteMap[j]].getStep();
      iter = ((ub - 1) - lb + loopStep) / loopStep;
      stride = loopStep * accessMatrix[accessMatrix.size() - 1][permuteMap[j]];
      bool isZero = true;
      for (unsigned i = 0; i < accessMatrix.size() - 1; i++) {
        // Check if the last col is zero until the last element.
        if (accessMatrix[i][permuteMap[j]] != 0)
          isZero = false;
      }
      if (isZero) {
        // Check what last elem is, If it's zero then give this column a score
        // of '1'. Else give this column a score of (iter/cls/(loopStep *
        // lastElem)).
        if (accessMatrix[accessMatrix.size() - 1][permuteMap[j]] == 0) {
          // If last element is zero and the col is last then cost is 1.
          if (j == (int)accessMatrix[0].size() - 1)
            cost *= 1;
          else
            // If col is not last then assuming that data from previous
            // iteration didn't fit in cache, There will be 'n' misses.
            cost *= iter;
        } else {
          // Last element is not zero hence some cache misses will be
          // encountered. If the 'stride' is less than 'cls' then some spatial
          // re-use is present.
          if (stride < cls) {
            cost *= ((iter / cls) / (stride));
          }
          // If not then no spatial reuse is present and all will be misses.
          else {
            cost *= iter;
          }
        }
      }
      // If the elements are not zero then we can stop here and conclude the
      // remaining cache misses are product of iterations of remaining loops.
      else {
        for (int l = j; l >= 0; l--) {
          lb = loopNest->loops[permuteMap[l]].getConstantLowerBound();
          ub = loopNest->loops[permuteMap[l]].getConstantUpperBound();
          loopStep = loopNest->loops[permuteMap[l]].getStep();
          iter = ((ub - 1) - lb + loopStep) / loopStep;
          cost *= iter;
        }
        break;
      }
    }
    loopNest->loadStoreInfo.loadsAndStores[refGroupInx]->getLoc().dump();
    std::cout << "access matrix cost: " << cost << "\n";
		totalCost += cost;
  }
	return totalCost;
}

void AffineLoopTransform::runOnFunction() {
  // Collect the loads and stores within the function.
  //  loadsAndStores.clear();
  // Find all Perfectly nested loops.
  getLoopNests(getFunction(), &perfectLoopNests);
  // Find all Imperfectly nested loops.
  // getLoopNests(getFunction(), &imperfectLoopNests);
  // Print all the Perfectly nested loops.
  /*
  for (auto loopNest : perfectLoopNests) {
    llvm::outs() << "Perfect loop nest at: " << loopNest.loops[0].getLoc()
                 << "\n";
    for (auto loop : loopNest.loops) {
      llvm::outs() << "Loop in nest at: " << loop.getLoc() << "\n";
      // Loop Attributes.
      llvm::outs() << "num results in maplowerbound: "
                   << loop.getLowerBound().getMap().getNumResults() << "\n";
      llvm::outs() << "num results in maplowerbound: "
                   << loop.getLowerBound().getMap().getNumResults() << "\n";
      if (loop.getLowerBound().getMap().getNumResults() > 1 ||
          loop.getUpperBound().getMap().getNumResults() > 1) {
        loopNest.isRectangular = false;
      }
    }
  }
        */
  // Find the loads/stores in loop nests.
  for (auto &loopNest : perfectLoopNests) {
    // check which is the parent for loop, once found walk that for loop.
    for (auto loop : loopNest.loops) {
      if (!isa<AffineForOp>(loop.getParentOp())) {
        // top level loop found.
        llvm::outs() << "Top level Loop: " << loop.getLoc() << "\n";
        loop.walk([&](Operation *op) {
          if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op)) {
            loopNest.loadStoreInfo.loadsAndStores.push_back(op);
          }
        });
      }
    }
  }

  // Print the loads/stores in loop nests.
  for (auto loopNest : perfectLoopNests) {
    for (auto loadOrStore : loopNest.loadStoreInfo.loadsAndStores) {
      llvm::outs() << "load/store at: " << loadOrStore->getLoc() << "\n";
    }
  }

  // Processing the loads and stores to get FlatAffineConsttraints.
  for (auto &loopNest : perfectLoopNests) {
    for (auto loadOrStore : loopNest.loadStoreInfo.loadsAndStores) {
      FlatAffineConstraints dependenceConstraints;

      auto *srcOpInst = loadOrStore;
      MemRefAccess srcAccess(srcOpInst);

      // Get composed access function for 'srcAccess'.
      AffineValueMap srcAccessMap;
      srcAccess.getAccessMap(&srcAccessMap);

      // Get iteration domain for the 'srcAccess' operation.
      FlatAffineConstraints srcDomain;
      if (failed(getInstIndexSet(srcAccess.opInst, &srcDomain))) {
      }
      // return DependenceResult::Failure;
      // llvm::outs() << "srcDomain.dump(): ";
      // srcDomain.dump();
      // Build dim and symbol position maps for each access from access operand
      // Value to position in merged constraint system.
      ValuePositionMap1 valuePosMap;

      buildDimAndSymbolPositionMaps1(srcDomain, srcAccessMap, &valuePosMap,
                                     &dependenceConstraints);
      initDependenceConstraints1(srcDomain, srcAccessMap, valuePosMap,
                                 &dependenceConstraints);

      //  assert(valuePosMap.getNumDims() ==
      //       srcDomain.getNumDimIds() + dstDomain.getNumDimIds());

      // Create memref access constraint by equating src/dst access functions.
      // Note that this check is conservative, and will fail in the future when
      // local variables for mod/div exprs are supported.
      if (failed(addMemRefAccessConstraints1(
              srcAccessMap, valuePosMap, &dependenceConstraints, &loopNest))) {
      }
      // return DependenceResult::Failure;
      /*
      llvm::outs()
          << "\nAccess Matrix start-----------------------------------------\n";

      dependenceConstraints.dump();

      llvm::outs()
          << "\nAccess Matrix End-----------------------------------------\n";
                        */
    }
  }
  // Try to print the access matrices.
  for (auto loopNest : perfectLoopNests) {
    for (auto B : loopNest.loadStoreInfo.B) {
      llvm::outs() << "access matrix:\n";
      for (unsigned i = 0; i < B.size(); i++) {
        for (unsigned j = 0; j < B[i].size(); j++) {
          llvm::outs() << B[i][j] << " ";
        }
        llvm::outs() << "\n";
      }
      llvm::outs() << "\n";
    }
    for (auto b : loopNest.loadStoreInfo.b) {
      llvm::outs() << "trailing matrix:\n";
      for (unsigned i = 0; i < b.size(); i++) {
        for (unsigned j = 0; j < b[i].size(); j++) {
          llvm::outs() << b[i][j] << " ";
        }
        llvm::outs() << "\n";
      }
      llvm::outs() << "\n";
    }
  }

  // Post-process matrix 'b' to contain only one vector.
  // Try to print the access matrices.
  for (auto &loopNest : perfectLoopNests) {
    for (auto &b : loopNest.loadStoreInfo.b) {
      SmallVector<int64_t, 8> toPush;
      for (unsigned i = 0; i < b.size(); i++) {
        for (unsigned j = 0; j < b[i].size(); j++) {
          toPush.push_back(b[i][j]);
        }
      }
      b.clear();
      b.push_back(toPush);
    }
  }
  // The access matrix has been constructed at this point we can now go on and
  // compute static information(which will not change because ofinterchanges),
  // such as rank, score of temporal reuse for each access.
/*
	for (auto loopNest : perfectLoopNests) {
    std::vector<double> permuteMap{1, 2, 3};
    for (unsigned i = 0; i < loopNest.loadStoreInfo.loadsAndStores.size();
         ++i) {
      // Copy the access matrix into a vector<vector<float>>
      std::vector<std::vector<double>> accessMatrix(
          loopNest.loadStoreInfo.B[i].size(),
          std::vector<double>(loopNest.loadStoreInfo.B[i][0].size()));
      for (unsigned j = 0; j < loopNest.loadStoreInfo.B[i].size(); ++j) {
        for (unsigned k = 0; k < loopNest.loadStoreInfo.B[i][j].size(); ++k) {
          accessMatrix[j][k] = (double)loopNest.loadStoreInfo.B[i][j][k];
        }
      }
      // Compute rank.
      loopNest.loadStoreInfo.ranks.push_back(computeRank(accessMatrix));
      std::cout << "Rank is: " << loopNest.loadStoreInfo.ranks[i] << "\n";
      loopNest.loadStoreInfo.temporalScores.push_back(
          computeTemporalScore(loopNest.loadStoreInfo.B[i]));
      // Computing score for temporal re-use.
      std::cout << "Temporal Score is: "
                << loopNest.loadStoreInfo.temporalScores[i] << "\n";
      // Computing score for Spatial re-use.
      std::cout << "Spatial Score is: "
                << computeSpatialScore(loopNest.loadStoreInfo.B[i], permuteMap)
                << "\n";
    }
  }
*/
  // TODO: add things for the computation of group spatial/temporal resuse.

  // Seems things needed are in hand so we can go on to compute the depepndence
  // loop wise now.
  for (auto &loopNest : perfectLoopNests) {
    // Calculate  dependences.
    checkDependences(&loopNest);
    std::cout << "Dependence matrix is:\n";
    // Print the dependence matrix corresponding to the loop nest.
    for (auto dependence : loopNest.loadStoreInfo.dependenceMatrix) {
      for (unsigned i = 0; i < dependence.size(); i++) {
        std::cout << dependence[i] << " ";
      }
      std::cout << "\n";
    }
  }

  // Now we have the dependence matrix, we can rule out the invalid interchanges
  // from the set of all valid dependenceies.
  for (auto &loopNest : perfectLoopNests) {
    // Generate all possible permutations.
    std::vector<int64_t> toGen;
    std::vector<std::vector<int64_t>> allPermutations;
    for (unsigned i = 0; i < loopNest.loops.size(); i++)
      toGen.push_back(i);
    do {
      allPermutations.push_back(toGen);
    } while (std::next_permutation(toGen.begin(), toGen.end()));

    // Eliminate all the impossibel dependences.
    eliminateInvalidDependence(&loopNest, allPermutations);
    std::cout << "no. of valid permutations: "
              << loopNest.loadStoreInfo.validPermuataions.size() << std::endl;
  }
  /*
          //std::cout<<"size before removal:
     "<<perfectLoopNests[0].loadStoreInfo.rarDependences.size()<<std::endl;
          // Go on and remove the spurious(with different acess matrices)
     dependeces from rarDependences. std::cout<<"dependence before removal\n";
          for(auto &loadOrStore:
     perfectLoopNests[0].loadStoreInfo.rarDependences){
                          //std::cout<<loadOrStore.srcOpInst<<"
     "<<loadOrStore.dstOpInst<<"\n"; std::cout<<"dependence cmp\n";
                  loadOrStore.srcOpInst->getLoc().dump();std::cout<<"\n";
                  loadOrStore.dstOpInst->getLoc().dump();std::cout<<"\n";
                  std::cout<<"dependence cmp\n";
          }
          std::cout<<"\n";
  */
  for (auto &loopNest : perfectLoopNests) {
    SmallVector<int, 4> toRemove;
    for (unsigned i = 0; i < loopNest.loadStoreInfo.rarDependences.size();
         i++) {
      // Find First access matrix.
      unsigned srcInx, dstInx;
      for (srcInx = 0; srcInx < loopNest.loadStoreInfo.loadsAndStores.size();
           srcInx++) {
        if (loopNest.loadStoreInfo.loadsAndStores[srcInx] ==
            loopNest.loadStoreInfo.rarDependences[i].srcOpInst)
          break;
      }
      for (dstInx = 0; dstInx < loopNest.loadStoreInfo.loadsAndStores.size();
           dstInx++) {
        if (loopNest.loadStoreInfo.loadsAndStores[dstInx] ==
            loopNest.loadStoreInfo.rarDependences[i].dstOpInst)
          break;
      }
      // Compare the access matrix of both the operands.
      if (!compareAccessMatrix(loopNest.loadStoreInfo.B[srcInx],
                               loopNest.loadStoreInfo.B[dstInx]))
        toRemove.push_back(i);
    }
    // Remove the bad dependences from the rarDependence.
    for (auto i = toRemove.rbegin(); i != toRemove.rend(); ++i) {
      loopNest.loadStoreInfo.rarDependences.erase(
          loopNest.loadStoreInfo.rarDependences.begin() + *i);
    }
    toRemove.clear();
  }
  /*
          std::cout<<"dependence afterremoval \n";
          for(auto &loadOrStore:
     perfectLoopNests[0].loadStoreInfo.rarDependences){
                          //std::cout<<loadOrStore.srcOpInst<<"
     "<<loadOrStore.dstOpInst<<"\n"; std::cout<<"dependence cmp\n";
                  loadOrStore.srcOpInst->getLoc().dump();std::cout<<"\n";
                  loadOrStore.dstOpInst->getLoc().dump();std::cout<<"\n";
                  std::cout<<"dependence cmp\n";
          }
          //std::cout<<"size after removal:
     "<<perfectLoopNests[0].loadStoreInfo.rarDependences.size()<<std::endl;
  */
  // I think the dependeces in RARDependence are pruned i can start with algo of
  // creating refGroups.
  for (auto &loopNest : perfectLoopNests) {
		for(auto perm : loopNest.loadStoreInfo.validPermuataions){
    createRefGroups(&loopNest, perm);
    computeCacheMisses(&loopNest, perm); std::cout<<"\n\n\n\n";
		}
  }

  // Was trying to print out the operands of the block of the operand.
  // Commenting out for now.
  /*
  getFunction().walk([&](Operation *op) {
          if (isa<AffineForOp>(op) || isa<AffineForOp>(op)){

                  }
  });
  */

  //  for (auto loopNest : perfectLoopNests) {
  //    for (auto loadOrStore : loopNest.loadsAndStores) {
  //      MemRefAccess access(loadOrStore);
  //      AffineValueMap accessMap;
  //      access.getAccessMap(&accessMap);
  //      AffineMap map = accessMap.getAffineMap();
  //			//ValuePositionMap valuePosMap;
  //      ArrayRef<Value> operands = accessMap.getOperands();
  //      std::vector<SmallVector<int64_t, 8>> flatExprs;
  //      FlatAffineConstraints localVarCst;
  //      getFlattenedAffineExprs(map, &flatExprs, &localVarCst);
  //			llvm::outs()<<"Operation found:
  //";loadOrStore->getLoc().dump();llvm::outs()<<"\n";
  //			/*
  //			llvm::outs()<<"operands: ";
  //			for(unsigned i = 0; i < operands.size(); i++){
  //				llvm::outs()<<
  //			}
  //			llvm::outs()<<"\n ";
  //			*/
  //
  //      for (unsigned i = 0; i < flatExprs.size(); i++) {
  //        llvm::outs() << "component: ";
  //				for (unsigned j = 0; j < flatExprs[i].size();
  // j++)
  //{
  //          llvm::outs() <<flatExprs[i][j] << " ";
  //        }
  //        llvm::outs() << "\n";
  //      }
  //      llvm::outs() << "\n";
  //}
  //}

  // Find dependences between the load/stores in loop nests.
  /*
        for (auto loopNest : perfectLoopNests) {
    checkDependences(loopNest.loadsAndStores);
  }
*/
  /*
// Print all the imperfectly nested loops.
for (auto loopNest : imperfectLoopNests) {
llvm::outs() << "Imperfect loop nest at: " << loopNest[0].getLoc() << "\n";
for (auto loop : loopNest)
llvm::outs() << "Imperfect loop in nest at: " << loop.getLoc() << "\n";
}
  */
  /*
    getFunction().walk([&](Operation *op) {
      if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op))
        loadsAndStores.push_back(op);
    });
    // Print out the loads/stores here.
    for (auto lors : loadsAndStores) {
      MemRefAccess access(lors);
      AffineValueMap srcAccessMap;
      access.getAccessMap(&srcAccessMap);
      for (auto i = (unsigned)0; i < srcAccessMap.getNumResults(); i++) {
        srcAccessMap.getResult(i).dump();
      }
      llvm::outs() << "\n";
      if (!access.isStore()) {
        lors->emitRemark("load found: ");
      }
    }
  }
  */
}

namespace mlir {
void registerAffineLoopTransform() {
  PassRegistration<AffineLoopTransform> pass(
      "affine-loop-transform", "Perform affine loop transformations, "
                               "optimizing locality and paralellism.");
}
} // namespace mlir
