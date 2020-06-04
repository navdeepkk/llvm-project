//===- AffineLoopTransform.cpp - Does affineloop transformations for paralellism
// and locality ------------------===//
#include "iostream"
#include "limits"
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
#include "unordered_map"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-loop-interchange"

using namespace mlir;

namespace {

struct AffineLoopTransform
    : public PassWrapper<AffineLoopTransform, FunctionPass> {
  typedef struct {
    // Encodes if the loop nest is rectangular or not.
    bool isRectangular = true;
    // Encodes info is the loop nest has IfElse or not.
    bool containsIfElse = false;
    // Encodes info of all loads and stores.
    typedef struct {
			// Captures all loads/stores in the loop nest.
      SmallVector<Operation *, 4> loadsAndStores;
			// Captues the access matrix.
      std::vector<std::vector<SmallVector<int64_t, 8>>> B;
			// Captures the trailing matrix.
      std::vector<std::vector<SmallVector<int64_t, 8>>> b;
			// Temporarily holds the refGroups when processing.
      std::vector<SmallVector<Operation *, 4>> refGroups;
			// Captures the info wether a loop is paralell or not.
      std::unordered_map<int64_t, bool> paralellLoops;
      // {perm:{allRefGroups:{refgroup :refGroupCost}:loopCost}}
      // A consolidated structure that holds all the permuataions, with respective
      // refGroups and their cost.
      std::vector<std::pair<
          std::pair<
              std::vector<int64_t>,
              std::vector<std::pair<SmallVector<Operation *, 4>, double>>>,
          double>>
          permRefGroups;
			
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
      // Captures the cache misses corresponding to the permutation.
      std::vector<std::pair<std::vector<int64_t>, double>> permScores;
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
    // Captures info of all loads/stores in a loop nest.
    LoadStoreInfo loadStoreInfo;
    // Encodes info of all loops in a loop nest.
    SmallVector<AffineForOp, 4> loops;
  } LoopInfo;
  // Captures all perfect loops present in the IR.
  std::vector<LoopInfo> perfectLoopNests;
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
  unsigned srcNumIds = srcMap.getNumDims() + srcMap.getNumSymbols();
  ArrayRef<Value> srcOperands = srcAccessMap.getOperands();

  std::vector<SmallVector<int64_t, 8>> srcFlatExprs;
  FlatAffineConstraints srcLocalVarCst;
  // Get flattened expressions for the source destination maps.
  if (failed(getFlattenedAffineExprs(srcMap, &srcFlatExprs, &srcLocalVarCst)))
    return failure();
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
    // Local terms.
    for (unsigned j = 0, e = srcNumLocalIds; j < e; j++)
      eq[newLocalIdOffset + j] = srcFlatExpr[srcNumIds + j];
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

static void
getLoopNests(FuncOp f, std::vector<AffineLoopTransform::LoopInfo> *loopNests) {
  auto getPerfectLoopNests = [&](AffineForOp root) {
    AffineLoopTransform::LoopInfo capturedLoops;
    SmallVector<AffineForOp, 4> loops;
    // Check if the loop nest was perfect.
    bool isPerfectlyNested;
    bool containsIfElse;
    bool hasImmediateLoadStore;

    // Walk the root and see if there is any nonRectangular loop.
    root.walk([&](Operation *op) {
      if (auto forOp = dyn_cast<AffineForOp>(op)) {
        if (!forOp.getConstantLowerBound() || !forOp.hasConstantUpperBound()) {
          // Simply return without capturing this loopnest and this loopnest
          // wont be processed.
          return;
        }
      }
    });

    while (1) {
      loops.clear();
      getPerfectlyNestedLoops(loops, root);
      isPerfectlyNested = true;
      containsIfElse = false;
      hasImmediateLoadStore = false;
      // Walk the last captured loop and check if it is perfect or not.
      AffineForOp innermostLoop = loops[loops.size() - 1];
      innermostLoop.walk([&](Operation *op) {
        if (isa<AffineForOp>(op) && op != innermostLoop) {
          isPerfectlyNested = false;
        } else if (isa<AffineIfOp>(op)) {
          // check for affineIfOps, if present return and donot capture this
          // loopnest.
          containsIfElse = true;
        } else if ((isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op)) &&
                   (op->getParentOp() == innermostLoop)) {
          hasImmediateLoadStore = true;
        }
      });
      // If it contains an if statement then truly return, this condtiotn means
      // this loop nest is not to be processed.
      if (containsIfElse || (!isPerfectlyNested && hasImmediateLoadStore))
        return;
      // If Loop is perfeclty nested then do nothing and break.
      if (isPerfectlyNested)
        break;
      // If the nest was imperfect try to make it perfect.
      if (!isPerfectlyNested) {
        // Initialize an opBuilder and make a new loop where imperfection is
        // found.
        OpBuilder opb(innermostLoop.getOperation()->getBlock(),
                      std::next(Block::iterator(innermostLoop.getOperation())));
        AffineForOp newLoop;
        newLoop = static_cast<AffineForOp>(opb.clone(*innermostLoop));
        // Once a new loop is made then try to eliminate the immediate forOp
        // children.
        unsigned int counter;
        unsigned int numForOps = 0;
        // Count number of affineForOps in the nest.
        innermostLoop.walk([&](Operation *op) {
          if (op->getParentOp() == innermostLoop && isa<AffineForOp>(op)) {
            ++numForOps;
          }
        });
        // Walk to remove the first half from the first nest.
        counter = 0;
        innermostLoop.walk([&](Operation *op) {
          if (op->getParentOp() == innermostLoop && isa<AffineForOp>(op)) {
            // Erase if alternate element.
            if (counter >= ceil(numForOps / 2) && (counter < numForOps)) {
              op->erase();
            }
            ++counter;
          }
        });
        // Walk to remove second half from second nest.
        counter = 0;
        newLoop.walk([&](Operation *op) {
          if (op->getParentOp() == newLoop && isa<AffineForOp>(op)) {
            // Erase if alternate element.
            if (counter < ceil(numForOps / 2)) {
              op->erase();
            }
            ++counter;
          }
        });
      }
    }
    // Push the loops found to the
    capturedLoops.loops = loops;
    loopNests->push_back(capturedLoops);
  };
  // Actual logic to make things work.
  for (auto &block : f)
    for (auto &op : block)
      if (auto forOp = dyn_cast<AffineForOp>(op))
        getPerfectLoopNests(forOp);
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

// Pushes the dependence into the appropriate structure according to the type of
// the dependence.
static void
buildDepVector(AffineLoopTransform::LoopInfo *loopNest,
               SmallVector<DependenceComponent, 2> dependenceComponents,
               Operation *srcOpInst, Operation *dstOpInst, bool isStatic) {

  loopNest->loadStoreInfo.insertIntoDependences(dependenceComponents, isStatic);
  // Compute the dependence vectors as needed.
  SmallVector<int64_t, 4> depVector;
  condenseDependences(dependenceComponents, &depVector);
  // Handle RAR dependences differently.
  // Don't push RAR into the dependence vector.
  if (isa<AffineLoadOp>(srcOpInst) && isa<AffineLoadOp>(dstOpInst)) {
    // Once the dependence is found insert it into a rarDependences
    // vector.
    loopNest->loadStoreInfo.insertIntoRarDependences(srcOpInst, dstOpInst,
                                                     depVector);
  } else {
    // Once the dependence is found insert it into a WrwDependences
    // vector.
    loopNest->loadStoreInfo.insertIntoWrwDependences(srcOpInst, dstOpInst,
                                                     depVector);
    if (!isStatic) {
      // If dependence is static push it into the dependenceMatrix.
      loopNest->loadStoreInfo.dependenceMatrix.push_back(depVector);
    }
  }
}

// Modifying to my needs. Passing in the reference of the loop nest which
// contains the loads/stores so that it is easy to push the dependences.
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

        // Can extract the access functions here from the returned
        // FlatAffineCostraints, irrespective of whether dependence is present
        // or not. The equalitites in the structure will eventually represent
        // the access functions. Let's print and see them.

        if (!ret) {
          // Dependence does not exists.
        } else if (dependenceComponents.empty() || d > numCommonLoops) {
          // Static dependence exists.
          buildDepVector(loopNest, dependenceComponents, srcOpInst, dstOpInst,
                         /*isStatic*/ true);
        } else {
          // Dynamic dependence exists.
          buildDepVector(loopNest, dependenceComponents, srcOpInst, dstOpInst,
                         /*isStatic*/ false);
        }
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
    // Check if the selected permutaion is invalid by checking if any
    // depenedence in dependenceMatrix becomes lexicographically less than zero.
    isValid = true;
    for (unsigned i = 0; i < dependenceMatrix.size(); ++i) {
      for (unsigned j = 0; j < dependenceMatrix[0].size(); ++j) {
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
  for (unsigned i = 0; i < srcMatrix.size(); ++i) {
    for (unsigned j = 0; j < srcMatrix[0].size(); ++j) {
      if (srcMatrix[i][j] != dstMatrix[i][j]) {
        isEqual = false;
      }
    }
  }
  return isEqual;
}
template <typename T>
static void checkForTemporalReuse(AffineLoopTransform::LoopInfo *loopNest,
                                  mlir::Operation *srcOpInst,
                                  mlir::Operation *dstOpInst, T depVector,
                                  std::vector<int64_t> permuteMap,
                                  bool &toGroup) {

  for (auto &dep : depVector) {
    // Solution exists for if true. We can then check if spatial or
    // temporal is present.
    if (((dep.srcOpInst == srcOpInst && dep.dstOpInst == dstOpInst) ||
         (dep.srcOpInst == dstOpInst && dep.dstOpInst == srcOpInst)) &&
        (srcOpInst != dstOpInst)) {
      // Check for temproal resue. if only varies in the last dimension
      // of the dependence only then we can say that temporal reuse is
      // present.
      bool isZero = true;
      for (unsigned i = 0; i < dep.dependence.size() - 1; ++i) {
        if (dep.dependence[permuteMap[i]] != 0)
          isZero = false;
      }
      // if all the elements are 0 uptil the last element then temporal
      // reuse is present.
      if (isZero) {
        // TODO: add check for the condition given in paper for temporal
        // reuse.
        // The condition essentially means that temporal reuse can even
        // be exploited if there is agap of atmost '2' between
        // iterations.
        if (abs(dep.dependence[permuteMap[dep.dependence.size() - 1]]) <= 2) {
          toGroup = true;
          break;
        }
      }
    }
  }
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
      isVisited[toContinue] = true;
      // Find the size of cache line size.
      auto loadOp = dyn_cast<AffineLoadOp>(
          loopNest->loadStoreInfo.loadsAndStores[toContinue]);
      auto storeOp = dyn_cast<AffineStoreOp>(
          loopNest->loadStoreInfo.loadsAndStores[toContinue]);
      unsigned width;
      if (loadOp) {
        width = loadOp.getMemRefType().getElementType().getIntOrFloatBitWidth();
      } else {
        width =
            storeOp.getMemRefType().getElementType().getIntOrFloatBitWidth();
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
          checkForTemporalReuse(loopNest, srcOpInst, dstOpInst,
                                loopNest->loadStoreInfo.rarDependences,
                                permuteMap, toGroup);
          // If we have found that the group can be made then break. and dont
          // check for WRW, and then check if theere is no dependence but still
          // there is spatial re-use because of the last dimension just varying
          // a little(less than 'cls').
          if (!toGroup) {
            checkForTemporalReuse(loopNest, srcOpInst, dstOpInst,
                                  loopNest->loadStoreInfo.wrwDependences,
                                  permuteMap, toGroup);
            // then checking in WRW.
            // Check here for the last condition where the access matrix are the
            // same, but there is no dependence but still we can pair the
            // elements because of spatial reuse being present. to check the
            // spatial reuse we needto check the 'b', if it is the same overall
            // and only the entry corresponding to the last loop varies by a
            // small amount(cls)
            if (!toGroup) {
              // We already know the access matrix are the same. just check the
              // last value in the matrix and see if the constants just vary by
              // a small constant.
              bool isSame = true;
              MemRefAccess srcAccess(srcOpInst);
              MemRefAccess dstAccess(dstOpInst);
              if (srcAccess.memref == dstAccess.memref) {
                std::vector<SmallVector<int64_t, 8>> b1 =
                    loopNest->loadStoreInfo.b[toContinue];
                std::vector<SmallVector<int64_t, 8>> b2 =
                    loopNest->loadStoreInfo.b[repInx];
                for (unsigned i = 0; i < b1.size(); ++i) {
                  for (unsigned j = 0; j < b1[i].size() - 1; ++j) {
                    if (b1[i][j] != b2[i][j])
                      isSame = false;
                  }
                }
                if (isSame) {
                  // Check if difference in last elements is less than the CLS.
                  if (abs(b1[0][b1[0].size() - 1] - b2[0][b2[0].size() - 1]) <
                      cls /*CLS*/) {
                    toGroup = true;
                    // std::cout << "grouping because of spatial reuse and "
                    //             "dependence is\n";
                  }
                }
              }
            }
          }
        }
        // If to group is set then the elements can be mergerd nto one group
        // erase from the vector adn insert into the refGroup and then break the
        // loop.
        if (toGroup) {
          // Find the RefGroup which has the element loadsAndStores[toContinue]
          // and delete it.
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
          break;
        }
      }
    }
  }
	/*
  std::cout << "permutation: \n";
  for (auto x : permuteMap)
    std::cout << x << " ";
  std::cout << std::endl;
  std::cout << "size after group creation: "
            << loopNest->loadStoreInfo.refGroups.size() << std::endl;
  for (auto refGroup : loopNest->loadStoreInfo.refGroups) {
    std::cout << "refgroup start:\n";
    for (auto ls : refGroup) {
      ls->getLoc().dump();
    }
    std::cout << "refgroup end:\n";
  }
	*/
}

static double computeCacheMisses(AffineLoopTransform::LoopInfo *loopNest,
                                 std::vector<int64_t> permuteMap) {
  // Iterate through the representative of the refGroups and start calculating
  // cache miss for each refGroup.
  double totalCost = 0.0f;
  double cost;
  int64_t lb, ub, stride, iter, loopStep;
  std::vector<std::pair<SmallVector<Operation *, 4>, double>> costVec;
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
    auto loadOp = dyn_cast<AffineLoadOp>(
        loopNest->loadStoreInfo.loadsAndStores[refGroupInx]);
    auto storeOp = dyn_cast<AffineStoreOp>(
        loopNest->loadStoreInfo.loadsAndStores[refGroupInx]);
    unsigned width;
    if (loadOp) {
      width = loadOp.getMemRefType().getElementType().getIntOrFloatBitWidth();
    } else {
      width = storeOp.getMemRefType().getElementType().getIntOrFloatBitWidth();
    }
    unsigned cls = 64 / (width / 8);
    // 1-D access matrix needs to be handeled sperately.
    if (accessMatrix.size() == 1) {
      int j;
      for (j = accessMatrix[0].size() - 1; j >= 0; j--) {
        lb = loopNest->loops[permuteMap[j]].getConstantLowerBound();
        ub = loopNest->loops[permuteMap[j]].getConstantUpperBound();
        loopStep = loopNest->loops[permuteMap[j]].getStep();
        iter = ((ub - 1) - lb + loopStep) / loopStep;
        stride =
            loopStep * accessMatrix[accessMatrix.size() - 1][permuteMap[j]];
        if (accessMatrix[0][permuteMap[j]] == 0) {
          // If last element is zero then cost is 1.
          cost *= 1;
        } else {
          // Last element is not zero hence some cache misses will be
          // encountered. If the 'stride' is less than 'cls' then some spatial
          // re-use is present.
          if (stride < cls) {
            cost *= ((iter / cls) / (stride));
            // break here because all the iterations after point will have
            // n-misses.
            break;
          }
          // If not then no spatial reuse is present and all will be misses.
          else {
            cost *= iter;
            // break here because all the iterations after point will have
            // n-misses.
            break;
          }
        }
      }
			// Compute trailing cache misses.	
      for (int l = j - 1; l >= 0; l--) {
        lb = loopNest->loops[permuteMap[l]].getConstantLowerBound();
        ub = loopNest->loops[permuteMap[l]].getConstantUpperBound();
        loopStep = loopNest->loops[permuteMap[l]].getStep();
        iter = ((ub - 1) - lb + loopStep) / loopStep;
        cost *= iter;
      }
      // Store this cost for future use.
      costVec.push_back(std::make_pair(refGroup, cost));
			/*
      std::cout << "permutaion: ";
      for (auto elem : permuteMap)
        std::cout << elem << " ";
      std::cout << "PermCost: " << cost << "\n";
			*/
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
      for (unsigned i = 0; i < accessMatrix.size() - 1; ++i) {
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
			//compute trailing cache misses.
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
	/*
    std::cout << "permutaion: ";
    for (auto elem : permuteMap)
      std::cout << elem << " ";
    std::cout << "PermCost: " << cost << "\n";

    std::cout << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                 "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                 "xxxxxxxxxxxx\n";
	*/
    // Store this cost for future use.
    costVec.push_back(std::make_pair(refGroup, cost));
    totalCost += cost;
  }
  // finally push the cost here.
  loopNest->loadStoreInfo.permRefGroups.push_back(
      make_pair(make_pair(permuteMap, costVec), totalCost));
  return totalCost;
}

static bool sortByVal(std::pair<std::vector<int64_t>, double> &a,
                      std::pair<std::vector<int64_t>, double> &b) {
  return (a.second < b.second);
}

static void findParalellLoops(AffineLoopTransform::LoopInfo *loopNest) {
  // Traverse the dependence matrix column-wise and check if the loop is
  // parallel. A Loop is paralell if it does not carry any dependence, i.e. all
  // entries in the col are 0.
  SmallVector<SmallVector<int64_t, 4>, 4> &dependenceMatrix =
      loopNest->loadStoreInfo.dependenceMatrix;
  unsigned short r = dependenceMatrix.size();
  // For the case when no dependence is present.
  if (r == 0) {
    for (unsigned i = 0; i < loopNest->loops.size(); ++i) {
      loopNest->loadStoreInfo.paralellLoops[i] = true;
    }
    return;
  }
  unsigned short c = dependenceMatrix[0].size();
  bool isParallel;
  for (auto j = 0; j < c; ++j) {
    isParallel = true;
    for (auto i = 0; i < r; ++i) {
      if (dependenceMatrix[i][j] != 0)
        isParallel = false;
    }
    loopNest->loadStoreInfo.paralellLoops[j] = isParallel;
  }
}

static void makePerm(AffineLoopTransform::LoopInfo *loopNest,
                     std::vector<int64_t> perm) {
  std::vector<unsigned int> permMap(perm.size());
  for (unsigned inx = 0; inx < perm.size(); ++inx) {
    permMap[perm[inx]] = inx;
  }
  permuteLoops(loopNest->loops, permMap);
}

void AffineLoopTransform::runOnFunction() {
  // Find all perfeclty nested loops. If the loops are not perfeclty nested
  // the call tries to make the loops perfect following a simple algorithm.
  getLoopNests(getFunction(), &perfectLoopNests);

  // Find the loads/stores in loop nests.
  for (auto &loopNest : perfectLoopNests) {
    // check which is the parent for loop, once found walk that for loop.
    for (auto loop : loopNest.loops) {
      if (!isa<AffineForOp>(loop.getParentOp())) {
        // top level loop found.
        loop.walk([&](Operation *op) {
          if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op)) {
            loopNest.loadStoreInfo.loadsAndStores.push_back(op);
          }
        });
      }
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
    }
  }
/*
  // Try to print the access matrices.
  for (auto loopNest : perfectLoopNests) {
    for (auto B : loopNest.loadStoreInfo.B) {
      llvm::outs() << "access matrix:\n";
      for (unsigned i = 0; i < B.size(); ++i) {
        for (unsigned j = 0; j < B[i].size(); ++j) {
          llvm::outs() << B[i][j] << " ";
        }
        llvm::outs() << "\n";
      }
      llvm::outs() << "\n";
    }
    for (auto b : loopNest.loadStoreInfo.b) {
      llvm::outs() << "trailing matrix:\n";
      for (unsigned i = 0; i < b.size(); ++i) {
        for (unsigned j = 0; j < b[i].size(); ++j) {
          llvm::outs() << b[i][j] << " ";
        }
        llvm::outs() << "\n";
      }
      llvm::outs() << "\n";
    }
  }
*/
  // Post-process matrix 'b' to contain only one vector.
  // Try to print the access matrices.
  for (auto &loopNest : perfectLoopNests) {
    for (auto &b : loopNest.loadStoreInfo.b) {
      SmallVector<int64_t, 8> toPush;
      for (unsigned i = 0; i < b.size(); ++i) {
        for (unsigned j = 0; j < b[i].size(); ++j) {
          toPush.push_back(b[i][j]);
        }
      }
      b.clear();
      b.push_back(toPush);
    }
  }

  // Seems things needed are in hand so we can go on to compute the dependence
  // loop wise now.
  for (auto &loopNest : perfectLoopNests) {
    // Calculate  dependences.
    checkDependences(&loopNest);
		/*
    std::cout << "Dependence matrix is:\n";
    // Print the dependence matrix corresponding to the loop nest.
    for (auto dependence : loopNest.loadStoreInfo.dependenceMatrix) {
      for (unsigned i = 0; i < dependence.size(); ++i) {
        std::cout << dependence[i] << " ";
      }
      std::cout << "\n";
    }
		*/
  }

  // Now we have the dependence matrix, we can rule out the invalid interchanges
  // from the set of all valid dependenceies.
  for (auto &loopNest : perfectLoopNests) {
    // Generate all possible permutations.
    std::vector<int64_t> toGen;
    std::vector<std::vector<int64_t>> allPermutations;
    for (unsigned i = 0; i < loopNest.loops.size(); ++i)
      toGen.push_back(i);
    do {
      allPermutations.push_back(toGen);
    } while (std::next_permutation(toGen.begin(), toGen.end()));

    // Eliminate all the invalid dependences.
    eliminateInvalidDependence(&loopNest, allPermutations);
    //std::cout << "no. of valid permutations: "
    //          << loopNest.loadStoreInfo.validPermuataions.size() << std::endl;
  }

	// Prune the unwanted RARdependences.
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
          loopNest.loadStoreInfo.rarDependences.begin() + (*i));
    }
    toRemove.clear();
  }

  // I think the dependeces in RARDependence are pruned, I can start with algo
  // of creating refGroups.
  for (auto &loopNest : perfectLoopNests) {
    for (auto perm : loopNest.loadStoreInfo.validPermuataions) {
      createRefGroups(&loopNest, perm);
      auto totalCost = computeCacheMisses(&loopNest, perm);
      //std::cout << "Total cost:" << totalCost;
      // Store scores in SmallVector and then sort them in the order of
      // inceasing cache misses.
      loopNest.loadStoreInfo.permScores.push_back(
          std::make_pair(perm, totalCost));
      std::sort(loopNest.loadStoreInfo.permScores.begin(),
                loopNest.loadStoreInfo.permScores.end(), sortByVal);
      //std::cout << "\n\n\n\n";
    }
  }
	
	/*
  // Just print the scores here. TODO: Remove this piece at last.
  for (auto &loopNest : perfectLoopNests) {
    for (auto elem : loopNest.loadStoreInfo.permScores) {
      std::cout << "perm: ";
      for (auto comp : elem.first) {
        std::cout << comp << " ";
      }
      std::cout << "second: " << elem.second << " ";
    }
    std::cout << "\n";
  }
	*/

  // Find all paralell loops in the loopNests.
  for (auto &loopNest : perfectLoopNests) {
    findParalellLoops(&loopNest);
  }
	/*
  // print all paralell loops in the loopNests.
  for (auto &loopNest : perfectLoopNests) {
    for (auto loop : loopNest.loadStoreInfo.paralellLoops) {
      std::cout << loop.first << " " << loop.second << "\n";
    }
  }
	*/

  // After parallel loops are found we can just find the best permutation.
  // Check here if more than one permutaitons with min cost is present. if yes,
  // then check if all the refgroups are the same or not.if yes, then check if
  // all the permutations have outer loop parallel. If yes, then choose the
  // permutaion among these which benifits the largest group, i.e. choose the
  // permutation which has the minimum cost for the largest group.

	/*
  for (auto &loopNest : perfectLoopNests) {
    for (auto perm : loopNest.loadStoreInfo.permRefGroups) {
      std::cout << "found permutatio: ";
      for (auto x : perm.first.first)
        std::cout << x << " ";
      std::cout << "----" << perm.second;
      std::cout << "\n";
    }
  }
	*/

  for (auto &loopNest : perfectLoopNests) {
    bool isPermuted = false;
    for (auto &perm : loopNest.loadStoreInfo.permScores) {
      //------------------------------------------------------------------------------------------------------------------------------------------//
      // check for fallback case.
      double minCost = perm.second;
      std::vector<std::pair<
          std::pair<
              std::vector<int64_t>,
              std::vector<std::pair<SmallVector<Operation *, 4>, double>>>,
          double>>
          minCostPerms;
      // Find the premutaions with min loop cost.
      for (auto refGroup : loopNest.loadStoreInfo.permRefGroups) {
        if (refGroup.second == minCost &&
            (loopNest.loadStoreInfo.paralellLoops[refGroup.first.first[0]] ==
             true)) {
					/*
          std::cout << "checking permutatio: ";
          for (auto x : refGroup.first.first)
            std::cout << x << " ";
          std::cout << "\n";
					*/
          // Adding only the permutations into the vectore which have cost as
          // mincost and have the outer loop paralell.
          minCostPerms.push_back(refGroup);
        }
      }
      // std::cout << "mincostpermize: " << minCostPerms.size() << "\n";
      if (minCostPerms.size() > 1) {
        std::vector<double> tieCost;
        double minCost = std::numeric_limits<double>::max();
        std::vector<int64_t> minCostPerm;
        // Find the score for each permutation.
        for (auto perm : minCostPerms) {
          double cost = 0;
          for (auto refGroup : perm.first.second) {
						// refgroupsize * (1 - (refgroupcost / totalcost))
            cost += (refGroup.first.size() *
                     (1.0f - (refGroup.second / perm.second)));
          }
          // store the tie breaking cost into the vector.
          tieCost.push_back(cost);
          if (cost < minCost) {
            minCostPerm = perm.first.first;
          }
        }
        // check if all the costs are same.
        bool areAllSame = false;
        if (std::adjacent_find(tieCost.begin(), tieCost.end(),
                               std::not_equal_to<>()) == tieCost.end()) {
          areAllSame = true;
        }
        // if all costs are same choose any refGroup, It does not matter.
        if (areAllSame == true) {
          // Choose the first permutation and permute.
          makePerm(&loopNest, minCostPerms[0].first.first);
          // std::cout << "permutin 1\n";
          isPermuted = true;
          break;
        } else {
          // take the minimum cost permutations and permute according to it.
          makePerm(&loopNest, minCostPerm);
          // std::cout << "permutin 2\n";
          isPermuted = true;
          break;
        }
      } else if (minCostPerms.size() == 1) {
        // take the minimum cost permutations and permute according to it.
        makePerm(&loopNest, minCostPerms[0].first.first);
        // std::cout << "permutin 3\n";
        isPermuted = true;
        break;
      }

      //-----------------------------------------------------------------------------------------------------------------------------------------//
      // Check if the loop at loopInx is parallel in the chosen perm.
      /*
            if (loopNest.loadStoreInfo.paralellLoops[perm.first[0]]) {
                                      std::vector<unsigned int>
         permMap(perm.first.size());
              // TODO: Verify if this approach is corrrect.
              // choose this permutation as the answer permute the loops and
         break.
              // First construct the permuteMap.
              for (unsigned inx = 0; inx < perm.first.size(); ++inx) {
                permMap[perm.first[inx]] = inx;
              }
              permuteLoops(loopNest.loops, permMap);
                                      std::cout<<"permutin 1\n";
              isPermuted = true;
              break;
            }*/
    }
    // No loop with outer paralellism or group benifit  was found simply change
    // to the best loop order in terms of cache misses.
    if (!isPermuted) {
      // Choose the first permutation as the interchange permutation.
      makePerm(&loopNest, loopNest.loadStoreInfo.permScores[0].first);
      // std::cout << "permutin 4\n";
    }
    //----------------------------------------------------------------------------------------------------------------------------------------//
  }
}

namespace mlir {
void registerAffineLoopTransform() {
  PassRegistration<AffineLoopTransform> pass(
      "affine-loop-interchange", "Perform affine loop transformations, "
                                 "optimizing locality and paralellism.");
}
} // namespace mlir
