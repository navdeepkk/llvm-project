//= TestSpecializeAffineForWMMA.cpp ----- =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Transforms/LoopUtils.h"

using namespace mlir;

#define DEBUG_TYPE "test-specialize-affine-matmul-for-wmma"

namespace {
struct TestSpecializeAffineForWMMA
    : public PassWrapper<TestSpecializeAffineForWMMA, FunctionPass> {
  void runOnFunction() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, mlir::vector::VectorDialect>();
  }

  // TODO: This should at the end be some kind of structure which has all the
  // operand numbers according to the version of the wmma op being used. Perhaps
  // it should be moved into a different header file.
  std::string ops[4] = {"AOp", "BOp", "COp", "DOp"};
  enum WmmaOps { AOp, BOp, COp, DOp };
  unsigned numElems[4] = {8, 8, 4, 4};
  constexpr static unsigned kMaxTiledLoops = 9;
  constexpr static unsigned kNumIntialLoops = 3;
  constexpr static unsigned kWMMAM = 16;
  constexpr static unsigned kWMMAN = 16;
  constexpr static unsigned kWMMAK = 16;
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
      SmallVector<Value> IVOperands;
      IVOperands.insert(IVOperands.end(), loop->getLowerBoundOperands().begin(),
                        loop->getLowerBoundOperands().end());
      IVOperands.insert(IVOperands.end(), loop->getUpperBoundOperands().begin(),
                        loop->getUpperBoundOperands().end());

      // The loops must be dependent from the outermost to the innermost loops.
      bool foundDependentLoopIV = false;
      for (auto operand : IVOperands) {
        // llvm::outs()<<"checking with "<<curMapStage<<"\n";
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

// Checks whether the given op can be hoisted by checking that
// - the op and any of its contained operations do not depend on SSA values
//   defined inside of the loop (by means of calling definedOutside).
// - the op has no side-effects. If sideEffecting is Never, sideeffects of
// this
//   op and its nested ops are ignored.
bool canBeHoisted(Operation *op, AffineForOp forOp,
                  SmallVector<AffineMap> &affineMaps,
                  SmallVector<SmallVector<Value>> &mapOprs) {
  //// Check that dependencies are defined outside of loop.
  // if (!llvm::all_of(op->getOperands(), definedOutside))
  //  return false;
  // Check whether this op is side-effect free. If we already know that there
  // can be no side-effects because the surrounding op has claimed so, we can
  // (and have to) skip this step.
  // bool isMovable = true;
  if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    if (auto mmaOp = dyn_cast<gpu::SubgroupMmaLoadMatrixOp>(op)) {
      // Check if the indices of the mmaLoadOp have any dependency to an affine
      // apply op.
      for (auto &operand : mmaOp->getOpOperands()) {
        if (auto defOp =
                dyn_cast<AffineApplyOp>(operand.get().getDefiningOp())) {
          // defOp.dump();
          AffineMap inxMap = defOp.getAffineMap();
          // llvm::outs()
          //    << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n";
          // inxMap.dump();
          SmallVector<Value> mapOpr(defOp.getMapOperands());
          fullyComposeAffineMapAndOperands(&inxMap, &mapOpr);
          canonicalizeMapAndOperands(&inxMap, &mapOpr);
          // llvm::outs()
          //    << "-------------------------------------------------------\n";
          // op->dump();
          // inxMap.dump();
          // for (auto op : mapOprs)
          //  op.dump();
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
    }
  }

  return true;
}

void getRecursiveUses(
    Operation *source, Operation *op, Operation *target,
    SmallVector<std::pair<Operation *, Operation *>> &loadStoreOps) {
  // llvm::outs()<<"called on\n";
  // op->dump();
  // target->dump();
  auto allUses = op->getUses();
  if (allUses.empty())
    return;
  for (auto &use : allUses) {
    if (use.getOwner() == target) {
      loadStoreOps.push_back(std::make_pair(source, target));
    } else {
      getRecursiveUses(source, use.getOwner(), target, loadStoreOps);
    }
  }
}

void moveLoopInvariantCode(AffineForOp forOp, OpBuilder &b) {
  auto &loopBody = forOp.getLoopBody();

  SmallVector<gpu::SubgroupMmaLoadMatrixOp> loadOps;
  SmallVector<gpu::SubgroupMmaStoreMatrixOp> storeOps;

  // Collect all the WMMAops in the body of the loop.
  for (auto &op : loopBody.getOps()) {
    if (auto mmaOp = dyn_cast<gpu::SubgroupMmaLoadMatrixOp>(op))
      loadOps.push_back(mmaOp);
    else if (auto mmaOp = dyn_cast<gpu::SubgroupMmaStoreMatrixOp>(op))
      storeOps.push_back(mmaOp);
  }

  // Construct pairs of load/store ops which access the same positon
  // in the same matrix. The implementaion is an O(n^3) approach in
  // worst case.
  SmallVector<std::pair<Operation *, Operation *>> loadStoreOps;
  for (auto loadOp : loadOps) {
    for (auto storeOp : storeOps) {
      getRecursiveUses(loadOp.getOperation(), loadOp.getOperation(),
                       storeOp.getOperation(), loadStoreOps);
    }
  }

  // for (auto p : loadStoreOps) {
  //  llvm::outs() << "-----------------------------------------------------\n";
  //  p.first->dump();
  //  p.second->dump();
  //}
  // Check wether an op in the pair is hoistable w.r.t to the surrounding loop.
  // llvm::outs() << loadStoreOps.size() << "\n";

  SmallVector<Value> newLoadOps;
  SmallVector<Operation *> movableOps;

  for (auto &p : loadStoreOps) {
    SmallVector<AffineMap> indexMaps;
    SmallVector<SmallVector<Value>> mapOprs;
    if (canBeHoisted(p.first, forOp, indexMaps, mapOprs)) {
      movableOps.push_back(p.first);
      // llvm::outs() << "hoistable op: ";
      // p.first->dump();
      // p.second->dump();

      // To move this pair of ops we need to to move the operands too. We have
      // already fetched the operands using the affine map composition and we
      // can safely create the same ops outside this for loop.
      b.setInsertionPoint(forOp);
      SmallVector<Value> indices;
      for (auto inx : llvm::zip(indexMaps, mapOprs)) {
        AffineMap affMap;
        SmallVector<Value> oprs;
        std::tie(affMap, oprs) = inx;
        indices.push_back(
            b.create<AffineApplyOp>(forOp.getLoc(), affMap, oprs));
      }

      // Create new ops. These ops will be used as iter_args for the forOp.
      auto origLoadop = cast<gpu::SubgroupMmaLoadMatrixOp>(p.first);
      newLoadOps.push_back(b.create<gpu::SubgroupMmaLoadMatrixOp>(
          forOp.getLoc(), origLoadop->getResultTypes()[0],
          origLoadop.srcMemref(), indices, origLoadop.leadDimension(),
          origLoadop.operand()));
    }
  }

  // Insert newly created ops as operands for the for op.
  SmallVector<Value> newOperands(forOp.getLowerBoundOperands());
  newOperands.append(forOp.getUpperBoundOperands().begin(),
                     forOp.getUpperBoundOperands().end());
  newOperands.append(newLoadOps);
  forOp->setOperands(newOperands);

  // Add newly created ops as arguments to the BB containing the loop body.
  SmallVector<BlockArgument> newArgs;
  for(auto newOp : newLoadOps){
    newArgs.push_back(loopBody.front().addArgument(newOp.getType()));
  }

  llvm::outs()<<"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n";
  llvm::outs() << forOp.getNumRegionIterArgs() << "\n";
  llvm::outs() << forOp.getNumIterOperands() << "\n";
  llvm::outs() << movableOps.size() << "\n";
  llvm::outs() << newOperands.size() << "\n";

  for(unsigned i = 0, e = movableOps.size(); i < e; ++i){
    movableOps[i]->getResult(0).replaceAllUsesWith(newArgs[i]);
  }

  // Set the newly created ops as iter_args for the forOp.
}

void TestSpecializeAffineForWMMA::runOnFunction() {
  // Try to find out the loop structure and identify the levels of tiling
  // done. Get the root for op first.
  FuncOp funcOp = getFunction();
  MLIRContext *context = funcOp.getContext();
  VectorType cOpType = VectorType::get(2, FloatType::getF16(context));
  // Only one AffineForOp expected, representing the root ForOp for the matmul
  // code.
  AffineForOp rootForOp;
  funcOp.walk([&](AffineForOp forOp) {
    if (!forOp->getParentOfType<AffineForOp>()) {
      rootForOp = forOp;
      WalkResult::interrupt();
    }
  });

  SmallVector<AffineForOp> computeLoops;
  findComputeLoops(rootForOp, computeLoops);

  // The expected number of loops 9 i.e., all matmul loops are tiled two
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
  // TODO: Add CL option to get WMMA size once more version of WMMA ops are
  // introcuced.
  computeLoops[2 * kNumIntialLoops].setStep(kWMMAM);
  computeLoops[2 * kNumIntialLoops + 1].setStep(kWMMAN);
  computeLoops[2 * kNumIntialLoops + 2].setStep(kWMMAK);

  // Now try to get the source/destination matrices for matmul by inspecting
  // the innermost loop body. We'll assume the first load to be the `A`
  // operand, second to be the `B` operand, Third to be the `c` operand.
  AffineForOp innermostLoop = computeLoops[computeLoops.size() - 1];
  Block &body = innermostLoop.getLoopBody().front();
  OpBuilder b(rootForOp.getContext());
  b.setInsertionPointAfter(innermostLoop);
  AffineForOp newInnermostLoop =
      b.create<AffineForOp>(rootForOp.getLoc(), 0, 0, innermostLoop.getStep());

  newInnermostLoop.setLowerBound(innermostLoop.getLowerBoundOperands(),
                                 innermostLoop.getLowerBoundMap());
  newInnermostLoop.setUpperBound(innermostLoop.getUpperBoundOperands(),
                                 innermostLoop.getUpperBoundMap());

  b.setInsertionPointToStart(&newInnermostLoop.getLoopBody().front());
  Location loc = rootForOp.getLoc();

  SmallVector<Value> wmmaOps;
  unsigned numOpsProcessed = 0;
  for (auto op = body.begin(), e = body.end(); op != e; ++op) {
    if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      assert(numOpsProcessed <= 2 &&
             "Recipe for tensor core matmul failed, "
             "innermost body doesn't represent a matmul");
      SmallVector<Value> index;
      SmallVector<Value> valueOperands;
      AffineMap opMap = loadOp.getAffineMap();
      auto operands = loadOp->getOpOperands().drop_front(1);

      // Get operands for the affine.apply op.
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

      MemRefType opType = loadOp.memref().getType().cast<MemRefType>();
      if (!opType.getAffineMaps().empty() &&
          !opType.getAffineMaps().front().isIdentity()) {
        // TODO: Handle such cases.
      } else {
        // Create GPU WMMA loadOp.
        wmmaOps.push_back(b.create<gpu::SubgroupMmaLoadMatrixOp>(
            loc,
            gpu::MMAFragmentType::get(numElems[numOpsProcessed % 3], cOpType),
            loadOp.memref(), index, b.getIndexAttr(opType.getDimSize(0)),
            b.getStringAttr(ops[numOpsProcessed % 3])));
        ++numOpsProcessed;
      }
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
      if (numOpsProcessed != 0 && numOpsProcessed % 3 == 0) {
        // If this is the last loadOp that is being processed, Then we can
        // emit the compute and store op also.
        wmmaOps.push_back(b.create<gpu::SubgroupMmaComputeOp>(
            loc, gpu::MMAFragmentType::get(numElems[COp], cOpType),
            wmmaOps[numOpsProcessed - 3], wmmaOps[numOpsProcessed - 2],
            wmmaOps[numOpsProcessed - 1]));

        MemRefType opType = storeOp.memref().getType().cast<MemRefType>();
        b.create<gpu::SubgroupMmaStoreMatrixOp>(
            loc, wmmaOps[numOpsProcessed], storeOp.memref(),
            cast<gpu::SubgroupMmaLoadMatrixOp>(
                wmmaOps[numOpsProcessed - 1].getDefiningOp())
                .indices(),
            b.getIndexAttr(opType.getDimSize(0)));
      }
    }
  }

  // Erase this for op and the newInnermostLoop at the correct position.
  innermostLoop.erase();
  computeLoops[computeLoops.size() - 1] = newInnermostLoop;

  // Sink down the K-loop to an inner level, just inside the warp space
  // `i` and `j` loops. The operations in `k` loop tile the warp space
  // `i` must be moved with it. This, makes copy loop execute more number
  // of times but, when these loops are mapped to a warp, It is ideally
  // expected that they have a single iteration. So the copies will actually
  // happen only once.
  b.setInsertionPointToStart(&computeLoops[4].getLoopBody().front());

  SmallVector<Operation *> toErase;
  Block &kBody = computeLoops[2].getLoopBody().front();

  // Gather all operations to erase between the global `k` loop and the
  // dimension and the warp-space loops. Clone them first just inside the
  // sunken `k` loop.
  for (auto op = kBody.begin(), e = kBody.end();
       op != e && &*op != computeLoops[3].getOperation(); ++op) {
    // TODO: The loop struture has been inspected and it has been checked that
    // only copy loops exist here.
    b.clone(*op);
    toErase.push_back(&*op);
  }

  // Erase all the gathered ops.
  for (auto op : toErase)
    op->erase();

  // Interchange the warp space loops with `k` loop.
  interchangeLoops(computeLoops[2], computeLoops[3]);
  interchangeLoops(computeLoops[2], computeLoops[4]);

  // Update positions in the original list.
  std::swap(computeLoops[3], computeLoops[2]);
  std::swap(computeLoops[4], computeLoops[3]);

  // Permute the innermost loop nest to bring `k` at the outermost position.
  MutableArrayRef<AffineForOp> toPermute(computeLoops.begin(),
                                         computeLoops.end());
  permuteLoops(toPermute.drop_front(6), {1, 2, 0});
  std::swap(computeLoops[8], computeLoops[7]);
  std::swap(computeLoops[6], computeLoops[7]);

  // Unroll-Jam the innermost `i` loop by factor equal to trip count.
  if (getConstantTripCount(computeLoops[7]).hasValue()) {
    loopUnrollJamByFactor(computeLoops[7],
                          getConstantTripCount(computeLoops[7]).getValue());
  }

  // Unroll the innermostLoop completely.
  loopUnrollFull(computeLoops[8]);

  // Promote the now innermostLoop, which is the `k` loop.
  // TODO: Check why failure is returned even when the loop has been promoted.
  if (promoteIfSingleIteration(computeLoops[6]).Failure)
    innermostLoop = computeLoops[5];
  else
    innermostLoop = computeLoops[6];

  // We need to move ops from inside to the outside level which are invariant
  // on the surrounding loop ivs. We handle side effecting operations in a
  // special way, if the side effecting operations are loop invariant than
  // they can be moved out. If the side effecting operations read and write to
  // the same location then they can still be moved out of the loops using
  // appropriate yield ops and also supplying loaded values back into the
  // invariant loop as iter_args. This would also require substituing the
  // usign values with the iter_arg.

  // Do not use walk here, as we do not want to go into nested regions and
  // hoist operations from there. These regions might have semantics unknown
  // to this rewriting. If the nested regions are loops, they will have been
  // processed.
  // funcOp->walk([&](AffineForOp forOp) { moveLoopInvariantCode(forOp, b); });
  moveLoopInvariantCode(innermostLoop, b);

  // llvm::DenseMap<Operation *, bool> isInnermostLoopOp;
  // Block &innermostLoopBody = innermostLoop.getLoopBody().front();

  // for (auto op = innermostLoopBody.begin(), e = innermostLoopBody.end();
  //     op != e; ++op) {
  //  isInnermostLoopOp[&*op] = false;
  //}

  // Move loop invariant code out of the innermostLoop
  // funcOp.walk(
  //    [&](AffineForOp forOp) { mlir::moveAffineLoopInvariantCode(forOp); });

  //// Mark all loops that were moved out.
  // for (auto op = innermostLoopBody.begin(), e = innermostLoopBody.end();
  //     op != e; ++op) {
  //  isInnermostLoopOp[&*op] = true;
  //}

  //// Gather all the operations in innermostLoop now to yeild them from the
  //// innermost to warp space `j` loop.
  // SmallVector<Value> newOperands;
  // for (auto op : isInnermostLoopOp) {
  //  if (op.second == false)
  //    if (auto wmmaOp = dyn_cast<gpu::SubgroupMmaLoadMatrixOp>(op.first))
  //      newOperands.push_back(wmmaOp);
  //}

  //// newOperands.push_back(computeLoops[5].getInductionVar());
  //// for (auto op = innermostLoopBody.begin(), e = innermostLoopBody.end();
  ////     op != e; ++op) {
  ////  if (auto mmaOp = dyn_cast<gpu::SubgroupMmaComputeOp>(op))
  ////    newOperands.push_back(mmaOp->getResult(0));
  ////}

  //// b.setInsertionPointToEnd(&computeLoops[4].getLoopBody().front());

  // llvm::outs() << newOperands.size() << "\n";
  //// computeLoops[4]->getParentOfType<FuncOp>().dump();
  // computeLoops[4]->setOperands(newOperands);
  // computeLoops[5]->setOperands(newOperands);

  // Get all the MMA compute ops inside the innermost loops. The result of
  // these ops are the ones that we want to yield and at the end WMMA store op
  // will consume these ops.
  // SmallVector<Value> toYield;
  // for (auto &op : innermostLoopBody) {
  //  if (auto wmmaOp = dyn_cast<gpu::SubgroupMmaComputeOp>(op))
  //    toYield.push_back(wmmaOp->getResult(0));
  //}
  // llvm::outs() << toYield.size() << "\n";

  // Set the innermostLoop to yeild the results of WMMA compute ops.

  // TODO: REMOVE: comfirming if operands have been set or not.
  // for (auto &ops : computeLoops[4]->getOpOperands())
  //  ops.get().getDefiningOp()->dump();
  // computeLoops[4]->getOperand(1).dump();
  // b.create<ConstantIndexOp>(loc, 3);
  // b.create<AffineYieldOp>(loc, toYield);
  // AffineYieldOp yieldOp =
  //    dyn_cast<AffineYieldOp>(computeLoops[5].getLoopBody().front().back());
  // yieldOp->setOperands(toYield);
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
