//== ---TestSpecializeAffineForWMMA.cpp - Test affine to GPU WMMA matmul -- ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains specilaization patterns for matmul targetting tensor cores
// on Nvidia GPUs.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/LoopUtils.h"

using namespace mlir;

#define DEBUG_TYPE "test-specialize-affine-matmul-for-wmma"

namespace {
struct TestSpecializeAffineForWMMA
    : public PassWrapper<TestSpecializeAffineForWMMA, FunctionPass> {
  void runOnFunction() override;

  TestSpecializeAffineForWMMA(){};
  TestSpecializeAffineForWMMA(const TestSpecializeAffineForWMMA &) {}
  explicit TestSpecializeAffineForWMMA(std::string accumulateType) {
    clAccumulateType = accumulateType;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, mlir::vector::VectorDialect>();
  }

  /// Order of loops required in the input IR in their relative order.
  enum LoopStructure {
    TbI,
    TbJ,
    TbK,
    WarpI,
    WarpJ,
    WarpK,
    ThreadI,
    ThreadJ,
    ThreadK
  };

  /// Number of total Matmul operands.
  constexpr static unsigned kNumMatmulOperands = 4;

  /// Enum containing GEMM operands, usefull in accessing certain containers.
  enum WmmaOps { AOp, BOp, COp, DOp };

  /// String Array representing the standard operands of matmul.
  std::string ops[4] = {"AOp", "BOp", "COp", "DOp"};

  /// Constant representing the maximum number of tiled loops that can be
  /// present in the input IR.
  constexpr static unsigned kMaxTiledLoops = 9;

  /// Constant representing the number of loops in untiled matmul.
  constexpr static unsigned kNumIntialLoops = 3;

  // TODO: This should be in some kind of structure which has all the operand
  // numbers according to the version of the wmma op being used. Perhaps
  // it should be moved into a different header file.
  /// Array representing the number of elements in the mmaFragment corresponding
  /// to a particular operand.
  unsigned numElemsToUse[kNumMatmulOperands],
      numElemsF16[kNumMatmulOperands] = {8, 8, 4, 4},
      numElemsF32[kNumMatmulOperands] = {8, 8, 8, 8};

  /// Constant representing the shape of WMMA op in M dimension.
  constexpr static unsigned kWMMAM = 16;

  /// Constant representing the shape of WMMA op in N dimension.
  constexpr static unsigned kWMMAN = 16;

  /// Constant representing the shape of WMMA op in K dimension.
  constexpr static unsigned kWMMAK = 16;

  Option<std::string> clAccumulateType{
      *this, "accum", llvm::cl::desc("Accumulate type to use for matmul."),
      llvm::cl::init("f16")};
};
} // end anonymous namespace

// Find out the Tile space loops. Three outermost loops are the tile space
// loops. Three loops are the minimum number of loops that are to be present
// in matrix multiplication. Since copy loops may also be present in the code,
// The input may not be perfectly nested. Assuming that the copy loops are
// annotated we can find differentiate them from the compute loops.
void findComputeLoops(AffineForOp rootForOp,
                      SmallVector<AffineForOp> &computeLoops) {
  bool nestedForExists = true;

  while (nestedForExists) {
    nestedForExists = false;
    computeLoops.push_back(rootForOp);
    // Scan for other for loops in the body which are not copy loops.
    Block &body = rootForOp.getLoopBody().front();

    for (auto op = body.begin(), e = body.end(); op != e; ++op) {
      if (auto forOp = dyn_cast<AffineForOp>(op)) {
        if (BoolAttr attr = forOp->getAttrOfType<BoolAttr>("isCopyLoopNest")) {
          // Make this forOp the next root.
          // TODO: Inset assertion for multiple non-copy loop children of this
          // for op.
          if (!attr.getValue()) {
            rootForOp = forOp;
            nestedForExists = true;
          }
        }
        rootForOp = forOp;
        nestedForExists = true;
      }
    }
  }
}

// Check that the loops are in the desired ordered, i.e.,
//		    Inter Thread-Block loops(i,j,k)
//		      Inter Warp loops(ii, jj, kk)
//			Intra Warp loops(iii, jjj, kkk)
void insepectTileStructure(SmallVector<AffineForOp> &computeLoops,
                           SmallVector<Value> &loopsIVs) {
  unsigned curMapStage = 0;
  for (auto loop = computeLoops.begin() +
                   TestSpecializeAffineForWMMA::kNumIntialLoops,
            e = computeLoops.end();
       loop < e; ++loop) {
    if (!loop->hasConstantBounds()) {
      // Insert lower/upper bound operands.
      SmallVector<Value> ivOperands;
      ivOperands.insert(ivOperands.end(), loop->getLowerBoundOperands().begin(),
                        loop->getLowerBoundOperands().end());
      ivOperands.insert(ivOperands.end(), loop->getUpperBoundOperands().begin(),
                        loop->getUpperBoundOperands().end());

      // The loops must be dependent from the outermost to the innermost loops.
      bool foundDependentLoopIV = false;
      for (auto operand : ivOperands) {
        if (operand == loopsIVs[curMapStage] ||
            operand == loopsIVs[curMapStage +
                                TestSpecializeAffineForWMMA::kNumIntialLoops])
          foundDependentLoopIV = true;
      }

      assert(
          foundDependentLoopIV == true &&
          "Recipe for tensor core matmul failed, improperly tiled loop nest");
      ++curMapStage;
      curMapStage %= TestSpecializeAffineForWMMA::kNumIntialLoops;
    }
  }
}

// Checks whether a given op is hoistable with respect to a forOp.
bool canBeHoisted(Operation *op, AffineForOp forOp,
                  SmallVector<AffineMap> &affineMaps,
                  SmallVector<SmallVector<Value>> &mapOprs) {
  auto isIndependentOfLoopIV = [&](MutableArrayRef<OpOperand> operands) {
    for (auto &operand : operands) {
      // TODO: Handle cases where the operands to the op may not be results of
      // AffineApplyOp.
      if (auto defOp = dyn_cast<AffineApplyOp>(operand.get().getDefiningOp())) {
        AffineMap inxMap = defOp.getAffineMap();
        SmallVector<Value> mapOpr(defOp.getMapOperands());
        fullyComposeAffineMapAndOperands(&inxMap, &mapOpr);
        canonicalizeMapAndOperands(&inxMap, &mapOpr);
        // After compostion check whether all the operands are independant of
        // the surrounding AffineForOp.
        if (llvm::all_of(mapOpr, [&](Value mapOpr) {
              return mapOpr != forOp.getInductionVar();
            })) {
          affineMaps.push_back(inxMap);
          mapOprs.push_back(mapOpr);
        } else
          return false;
      }
    }
    return true;
  };

  if (auto mmaOp = dyn_cast<gpu::SubgroupMmaLoadMatrixOp>(op)) {
    // Check if the indices of the mmaLoadOp have any dependency to an affine
    // apply op.
    return isIndependentOfLoopIV(mmaOp->getOpOperands());
  }

  return false;
}

// Fetches all the uses of an op until the use converges into an op
// with no results.
void getRecursiveUses(
    Operation *source, Operation *op, Operation *target,
    SmallVector<std::pair<Operation *, Operation *>> &loadStoreOps) {
  auto allUses = op->getUses();
  if (allUses.empty())
    return;
  for (auto &use : allUses) {
    // Inspect ops wihtout any regions, i.e., avoid forops, ifops etc.
    if (use.getOwner()->getNumRegions() == 0) {
      if (use.getOwner() == target) {
        loadStoreOps.push_back(std::make_pair(source, target));
      } else {
        getRecursiveUses(source, use.getOwner(), target, loadStoreOps);
      }
    }
  }
}

// Find all pairs of load/store ops that can be moved and move them just
// before/after the forOp.
void findAndMoveLoadStorePairs(
    AffineForOp forOp, OpBuilder &b,
    SmallVector<std::pair<Operation *, Operation *>> &loadStoreOps) {
  auto &loopBody = forOp.getLoopBody();

  SmallVector<gpu::SubgroupMmaLoadMatrixOp> loadOps;
  SmallVector<gpu::SubgroupMmaStoreMatrixOp> storeOps;

  // Collect all the WMMALoad/StoreOps in the body of the loop.
  for (auto &op : loopBody.getOps()) {
    if (auto mmaOp = dyn_cast<gpu::SubgroupMmaLoadMatrixOp>(op))
      loadOps.push_back(mmaOp);
    else if (auto mmaOp = dyn_cast<gpu::SubgroupMmaStoreMatrixOp>(op))
      storeOps.push_back(mmaOp);
  }

  // Find pairs of load/stores such that the value being stored is somehow
  // dependent on the load.
  for (auto loadOp : loadOps) {
    for (auto storeOp : storeOps) {
      getRecursiveUses(loadOp.getOperation(), loadOp.getOperation(),
                       storeOp.getOperation(), loadStoreOps);
    }
  }

  // If no load/store pair found, then return.
  if (loadStoreOps.size() == 0)
    return;

  SmallVector<Value> newLoadOps;
  SmallVector<Operation *> newStoreOps;
  SmallVector<Operation *> movableOps;
  SmallVector<SmallVector<Value>> newIndices;

  // Check if the load/store op pairs are hoistable.
  for (auto &p : loadStoreOps) {
    SmallVector<AffineMap> indexMaps;
    SmallVector<SmallVector<Value>> mapOprs;
    // TODO: Insert check for storeOp also.
    if (canBeHoisted(p.first, forOp, indexMaps, mapOprs)) {
      movableOps.push_back(p.first);

      b.setInsertionPoint(forOp);
      SmallVector<Value> indices;
      for (auto inx : llvm::zip(indexMaps, mapOprs)) {
        AffineMap affMap;
        SmallVector<Value> oprs;
        std::tie(affMap, oprs) = inx;
        indices.push_back(
            b.create<AffineApplyOp>(forOp.getLoc(), affMap, oprs));
      }

      // Store these new indices for use later, while moving the store ops.
      newIndices.push_back(indices);

      // To move this pair of ops we need to to move the operands too. We have
      // already fetched the operands using the affine map composition and we
      // can safely create the same ops outside this for loop. Create new load
      // ops. These ops will be used as iter_args for the forOp.
      auto origLoadop = cast<gpu::SubgroupMmaLoadMatrixOp>(p.first);
      newLoadOps.push_back(b.create<gpu::SubgroupMmaLoadMatrixOp>(
          forOp.getLoc(), origLoadop->getResultTypes()[0],
          origLoadop.srcMemref(), indices, origLoadop.leadDimension(),
          origLoadop.operand()));
    }
  }

  if (movableOps.size() == 0)
    return;

  // Insert newly created ops as operands for the for op.
  SmallVector<Value> newOperands(forOp.getLowerBoundOperands());
  newOperands.append(forOp.getUpperBoundOperands().begin(),
                     forOp.getUpperBoundOperands().end());
  newOperands.append(newLoadOps);
  forOp->setOperands(newOperands);

  // Add newly created ops as arguments to the basic block containing the loop
  // body.
  SmallVector<BlockArgument> newArgs;
  for (auto newOp : newLoadOps) {
    newArgs.push_back(loopBody.front().addArgument(newOp.getType()));
  }

  // Set the newly created ops as iter_args for the forOp.
  for (unsigned i = 0, e = movableOps.size(); i < e; ++i) {
    movableOps[i]->getResult(0).replaceAllUsesWith(newArgs[i]);
  }

  // Create a new affine forOp with body and clone the ops from the original
  // nest to this loop and then erase the original nest.
  b.setInsertionPointAfter(forOp);
  AffineForOp newForop = b.create<AffineForOp>(
      forOp.getLoc(), forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
      forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(), forOp.getStep(),
      forOp.getIterOperands(),
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
        builder.create<AffineYieldOp>(loc, iterArgs);
      });

  // Clone the body of the original forop into the newly create for op. First
  // add the iterArgs and loopIV into the clonigMap.
  BlockAndValueMapping mapping;
  mapping.map(forOp.getInductionVar(), newForop.getInductionVar());
  mapping.map(forOp.getRegionIterArgs(), newForop.getRegionIterArgs());

  b.setInsertionPointToStart(&newForop.getLoopBody().front());

  // Create storeOps just after the forOp. The operands to these store ops
  // will be the results yielded by the forOp.
  SmallVector<gpu::SubgroupMmaStoreMatrixOp> clonedStoreOps;
  for (auto &op : forOp.getLoopBody().front().without_terminator()) {
    Operation *clonedOp = b.clone(op, mapping);
    if (auto storeOp = dyn_cast<gpu::SubgroupMmaStoreMatrixOp>(clonedOp))
      clonedStoreOps.push_back(storeOp);
  }

  // Erase the original for op.
  forOp.erase();

  // Set the correct operands for the yield op.
  SmallVector<Value> toYield;
  AffineYieldOp yieldOp =
      dyn_cast<AffineYieldOp>(newForop.getLoopBody().front().back());

  for (auto op : clonedStoreOps) {
    toYield.push_back(op.src());
  }

  yieldOp->setOperands(toYield);

  // Place newly created storeOps just outside the for ops body and set their
  // operands to be the ops yeileded by the newly created AffineForOp.
  SmallVector<Value> newForOpRes(newForop.getResults());
  b.setInsertionPointAfter(newForop);
  for (auto resSrc : llvm::zip(newForOpRes, clonedStoreOps, newIndices)) {
    Value newRes;
    gpu::SubgroupMmaStoreMatrixOp clonedStoreOp;
    SmallVector<Value> indices;
    std::tie(newRes, clonedStoreOp, indices) = resSrc;
    newStoreOps.push_back(b.create<gpu::SubgroupMmaStoreMatrixOp>(
        newForop.getLoc(), newRes, clonedStoreOp.dstMemref(), indices,
        clonedStoreOp.leadDimension()));
    clonedStoreOp.erase();
  }

  // Update loadStoreOps to contain newly created load/store ops. Newly created
  // load/store ops are always candidates for further movement.
  loadStoreOps.clear();
  for (auto lSPair : llvm::zip(newLoadOps, newStoreOps)) {
    Value load;
    Operation *store;
    std::tie(load, store) = lSPair;
    loadStoreOps.push_back(std::make_pair(load.getDefiningOp(), store));
  }
}

// Utility to move invariant load/store pairs with respect to a forOp
// before/after respectively. The utility finds loads such that the result of
// the store is somehow dependent on the what was loaded. The loads are moved
// just before the forOp and the load results are set as iter_args for the
// forOp. The result of the computation is then yielded by the forOp and result
// of the forOp is then store by the storeOps which are moved just after the
// forOp. In this way redundant load/stores can be moved out of a given forOp.
void moveInvariantLoadStorePairs(FuncOp funcOp, OpBuilder b) {
  SmallVector<std::pair<Operation *, Operation *>> loadStoreOps;
  funcOp->walk([&](AffineForOp forOp) {
    findAndMoveLoadStorePairs(forOp, b, loadStoreOps);
  });
}

void TestSpecializeAffineForWMMA::runOnFunction() {
  FuncOp funcOp = getFunction();
  MLIRContext *context = funcOp.getContext();

  Type cdOpType, abOpType;
  abOpType = VectorType::get(2, FloatType::getF16(context));

  Type operandElemTypes[kNumMatmulOperands];

  // Initialize the numElems array to hold the correct number of elements by
  // inspecting the accumulate version.
  if (clAccumulateType.getValue().compare("f16") == 0) {
    cdOpType = abOpType;
    // Set number of operands for GPU WmmaOps.
    for (unsigned i = 0; i < kNumMatmulOperands; ++i)
      numElemsToUse[i] = numElemsF16[i];
  } else if (clAccumulateType.getValue().compare("f32") == 0) {
    cdOpType = FloatType::getF32(context);
    // Set number of operands for gpu WMMA op.
    for (unsigned i = 0; i < kNumMatmulOperands; ++i)
      numElemsToUse[i] = numElemsF32[i];
  }

  // Set Element type for GPU WmmaOps.
  operandElemTypes[AOp] = abOpType;
  operandElemTypes[BOp] = abOpType;
  operandElemTypes[COp] = cdOpType;
  operandElemTypes[DOp] = cdOpType;

  // Check and mark the parallel loops in the IR.
  funcOp.walk([&](AffineForOp loop) {
    if (isLoopParallel(loop))
      loop->setAttr("isParallel", BoolAttr::get(loop.getContext(), true));
  });


  // Get the root for op first.
  AffineForOp rootForOp;
  funcOp.walk([&](AffineForOp forOp) {
    if (!forOp->getParentOfType<AffineForOp>()) {
      rootForOp = forOp;
      WalkResult::interrupt();
    }
  });

  // Find All the compute loops in the rootForOp.
  SmallVector<AffineForOp> computeLoops;
  findComputeLoops(rootForOp, computeLoops);

  // The expected number of loops is 9 i.e., all matmul loops are tiled two
  // times.
  // TODO: Add cases when all the loops are not tiled.
  assert(computeLoops.size() == kMaxTiledLoops &&
         "Recipe for tensor core matmul failed, improperly tiled loop nest");

  // Find the different type of loops. When mapped to GPU there may be three
  // different types of loops present. 1.) Inter ThreadBlock-tile loops, 2.)
  // Inter Warp-tile loops 3.) Intra Warp-tile loops.
  SmallVector<Value> loopsIVs;
  for (auto loop : computeLoops) {
    loopsIVs.push_back(loop.getInductionVar());
  }

  // Check the tiling order of loops.
  insepectTileStructure(computeLoops, loopsIVs);

  // Insert GPU MMA ops in the innermost loop nest. This involves changing the
  // loop steps of the surrounding loops. To the size of WMMA operation and
  // then caluclating the right indices for load/store operations and also
  // identifying the leading dimensions of the source/destination memrefs.
  // First change the loop steps to MMA size.
  // TODO: Add CL option to get WMMA size once more versions of WMMA ops are
  // introcuced.
  computeLoops[ThreadI].setStep(kWMMAM);
  computeLoops[ThreadJ].setStep(kWMMAN);
  computeLoops[ThreadK].setStep(kWMMAK);

  // Create a new loop which will be the innermostLoop. This loop will have
  // gpuSubgroup WmmaOps instead of scalar loads/stores and compute ops.
  AffineForOp innermostLoop = computeLoops[ThreadK];
  Block &body = innermostLoop.getLoopBody().front();
  OpBuilder b(rootForOp.getContext());

  b.setInsertionPointAfter(innermostLoop);
  AffineForOp newInnermostLoop =
      b.create<AffineForOp>(rootForOp.getLoc(), 0, 0, innermostLoop.getStep());

  // Copy bounds from the original innermostLoop.
  newInnermostLoop.setLowerBound(innermostLoop.getLowerBoundOperands(),
                                 innermostLoop.getLowerBoundMap());
  newInnermostLoop.setUpperBound(innermostLoop.getUpperBoundOperands(),
                                 innermostLoop.getUpperBoundMap());

  // Insert wmmaOps in the newly created loop.
  b.setInsertionPointToStart(&newInnermostLoop.getLoopBody().front());
  Location loc = rootForOp.getLoc();
  SmallVector<Value> wmmaOps;
  unsigned numOpsProcessed = 0;

  // Helper to emit indices as affine.apply's for a load/store op.
  auto emitIndices = [&](SmallVector<Value> &index,
                         SmallVector<Value> &valueOperands, AffineMap &opMap,
                         MutableArrayRef<OpOperand> &operands) {
    for (auto operand = operands.begin(), e = operands.end(); operand < e;
         ++operand) {
      if (operand->get() == innermostLoop.getInductionVar()) {
        valueOperands.push_back(newInnermostLoop.getInductionVar());
      } else {
        valueOperands.push_back(operand->get());
      }
    }

    // Emit affine.apply's for each result expr in the map.
    for (unsigned i = 0, e = opMap.getNumResults(); i < e; ++i) {
      index.push_back(
          b.create<AffineApplyOp>(loc, opMap.getSubMap(i), valueOperands));
    }
  };

  // Now try to get the source/destination matrices for matmul in the original
  // innermostLoop and create corresponding ops in the new innermostLoop. We do
  // so by inspecting the innermost loop body. We'll assume the first load to be
  // the `A` operand, second to be the `B` operand, Third to be the `C` operand.
  // We also emit affine.apply's for each index of the load/store op in order to
  // use them as indices for gpuWmmaOps.
  for (auto op = body.begin(), e = body.end(); op != e; ++op) {
    SmallVector<Value> index;
    SmallVector<Value> valueOperands;
    AffineMap opMap;
    MutableArrayRef<OpOperand> operands;
    if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      assert(numOpsProcessed <= 2 &&
             "Recipe for tensor core matmul failed, "
             "innermost body probably doesn't represent a matmul");
      opMap = loadOp.getAffineMap();
      operands = loadOp->getOpOperands().drop_front(1);

      // Emit affine.apply's for each index.
      emitIndices(index, valueOperands, opMap, operands);
      // Cases when higher dimensional memrefs are present are unimplemented.
      if (index.size() != 2)
        return;

      MemRefType opType = loadOp.memref().getType().cast<MemRefType>();
      if (!opType.getAffineMaps().empty() &&
          !opType.getAffineMaps().front().isIdentity()) {
        // TODO: Handle such cases.
      } else {
        // Create GPU WMMA loadOp.
        wmmaOps.push_back(b.create<gpu::SubgroupMmaLoadMatrixOp>(
            loc,
            gpu::MMAFragmentType::get(numElemsToUse[numOpsProcessed],
                                      operandElemTypes[numOpsProcessed]),
            loadOp.memref(), index, b.getIndexAttr(opType.getDimSize(0)),
            b.getStringAttr(ops[numOpsProcessed])));
        ++numOpsProcessed;
      }
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
      if (numOpsProcessed == 3) {
        // If this is the last loadOp that is being processed, Then we can
        // emit the compute and store op also.
        wmmaOps.push_back(b.create<gpu::SubgroupMmaComputeOp>(
            loc,
            gpu::MMAFragmentType::get(numElemsToUse[COp],
                                      operandElemTypes[numOpsProcessed]),
            wmmaOps[numOpsProcessed - 3], wmmaOps[numOpsProcessed - 2],
            wmmaOps[numOpsProcessed - 1]));

        opMap = storeOp.getAffineMap();
        operands = storeOp->getOpOperands().drop_front(2);

        // Emit affine.apply's for each index.
        emitIndices(index, valueOperands, opMap, operands);

        MemRefType opType = storeOp.memref().getType().cast<MemRefType>();
        b.create<gpu::SubgroupMmaStoreMatrixOp>(
            loc, wmmaOps[numOpsProcessed], storeOp.memref(), index,
            b.getIndexAttr(opType.getDimSize(0)));
      }
    }
  }

  // Erase this for op and the newInnermostLoop at the correct position.
  innermostLoop.erase();
  computeLoops[ThreadK] = newInnermostLoop;

  // Sink down the K-loop to an inner level, just inside the warp space `i` and
  // `j` loops. The operations in `k` loop, before the warp space `i`, must be
  // moved with it. This, makes those operations get executed more number of
  // times than before, but, when these loops are mapped to a warp in GPU, It is
  // ideally expected that they have a single iteration. So They will get
  // executed only once.
  b.setInsertionPointToStart(&computeLoops[WarpJ].getLoopBody().front());

  SmallVector<Operation *> toErase;
  Block &kBody = computeLoops[TbK].getLoopBody().front();

  // Gather all operations between the global `k` loop and the warp-space loops.
  // Clone them just inside the sunken `k` loop.
  for (auto op = kBody.begin(), e = kBody.end();
       op != e && &*op != computeLoops[WarpI].getOperation(); ++op) {
    // TODO: Inspect the loop structure and guarantee that only copy loops exist
    // here.
    b.clone(*op);
    toErase.push_back(&*op);
  }

  // Erase all the gathered ops.
  for (auto *op : toErase)
    op->erase();

  // Interchange the warp space loops with `k` loop.
  interchangeLoops(computeLoops[TbK], computeLoops[WarpI]);
  interchangeLoops(computeLoops[TbK], computeLoops[WarpJ]);

  // Update positions in the original list.
  std::swap(computeLoops[WarpI], computeLoops[TbK]);
  std::swap(computeLoops[WarpK], computeLoops[WarpI]);

  // Permute the innermost loop nest to bring `k` at the outermost position.
  MutableArrayRef<AffineForOp> toPermute(computeLoops.begin(),
                                         computeLoops.end());

  permuteLoops(toPermute.drop_front(6), {1, 2, 0});

  // Update positions in the original list.
  std::swap(computeLoops[ThreadK], computeLoops[ThreadJ]);
  std::swap(computeLoops[ThreadI], computeLoops[ThreadJ]);

  // Unroll-Jam the innermost `i` loop by factor equal to trip count.
  if (getConstantTripCount(computeLoops[ThreadJ]).hasValue()) {
    assert(loopUnrollJamByFactor(
               computeLoops[ThreadJ],
               getConstantTripCount(computeLoops[ThreadJ]).getValue())
               .succeeded() &&
           "Unable to unroll loop, please inspect the IR");
  }

  // Unroll the innermostLoop completely.
  assert(loopUnrollFull(computeLoops[ThreadK]).succeeded() &&
         "Unable to unroll loop, please inspect the IR");

  // Promote the now innermostLoop, which is the `k` loop.
  // TODO: Check why failure is returned even when the loop has been promoted.
  if (promoteIfSingleIteration(computeLoops[ThreadI]).failed())
    innermostLoop = computeLoops[WarpK];
  else
    innermostLoop = computeLoops[ThreadI];

  // We need to move ops from inside to the outside level which are invariant
  // on the surrounding loop ivs. We handle side effecting operations in a
  // special way, if the side effecting operations are loop invariant than
  // they can be moved out. If the side effecting operations read and write to
  // the same location then they can still be moved out of the loops using
  // appropriate yield ops and also supplying loaded values back into the
  // invariant loop as iter_args. This would also require substituing the
  // usign values with the iter_arg.
  moveInvariantLoadStorePairs(funcOp, b);

  // Convert the marked forOps to parallel.
  // TODO: Currently this works because the parallel loops really doesn't yield
  // anything and affineParallelize also does not handle the case when the ops
  // yield somehing. Whenever this breaks we need to handle this.
  funcOp.walk([&](AffineForOp forOp) {
    if (auto isParallel = forOp->getAttrOfType<BoolAttr>("isParallel")) {
      if (isParallel.getValue() == true)
        affineParallelize(forOp);
    }
  });
}

namespace mlir {
namespace test {
void registerTestSpecializeAffineForWMMAPass() {
  PassRegistration<TestSpecializeAffineForWMMA>(
      "test-specialize-affine-matmul-for-wmma",
      "specialize affine matmul loops to use GPU WMMA ops");
}
} // namespace test
} // namespace mlir

