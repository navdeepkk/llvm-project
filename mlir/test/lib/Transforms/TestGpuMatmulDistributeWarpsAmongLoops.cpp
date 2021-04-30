//===------ TestGpuMatmulDistributeWarpsAmongLoops.cpp
//----------------------===//
//--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that distributes copy loops among warps for
// matmul.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/ParallelLoopMapper.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "distribute-warps-among-loops"

using namespace mlir;
using namespace mlir::scf;

namespace {
struct TestGpuMatmulDistributeWarpsAmongLoops
    : public PassWrapper<TestGpuMatmulDistributeWarpsAmongLoops,
                         OperationPass<ModuleOp>> {

  TestGpuMatmulDistributeWarpsAmongLoops() = default;
  TestGpuMatmulDistributeWarpsAmongLoops(
      const TestGpuMatmulDistributeWarpsAmongLoops &pass) {}

  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect>();
    registry.insert<gpu::GPUDialect>();
  }

  Option<int64_t> splitIndex{
      *this, "split-index",
      llvm::cl::desc("Specifies at which index to split the warps among loops"),
      llvm::cl::init(-1)};
};
} // end namespace

/// Checks if `loop` is a copy loop/loop nest or not. Here we have taken a
/// conservative approach for identifying copy loop. We define a loop as a
/// copy loop if it consists of exactly one load op and one store op.
bool checkIfCopyLoop(ForOp loop) {
  unsigned numLoad = 0, numStore = 0;
  loop.walk([&](LoadOp load) {
    if (load)
      numLoad++;
  });
  loop.walk([&](StoreOp store) {
    if (store)
      numStore++;
  });
  if (numLoad == 1 && numStore == 1)
    return true;
  return false;
}

/// Moves the ForOp at the beginning of IfOp.
template <typename OpTy>
static void moveBody(OpTy srcLoop, IfOp destIfOp) {
  Block *destBody = destIfOp.getBody();
  destBody->getOperations().splice(
      std::prev(destBody->end()),
      srcLoop.getOperation()->getBlock()->getOperations(),
      srcLoop.getOperation());
}

/// Distributes warps equally between the two set of copy loop nests if
/// pipelining using double buffering is done.
static void distributeWarpsAmongCopyLoops(gpu::LaunchOp launchOp,
                                          int64_t splitIndex) {
  SmallVector<ForOp, 2> copyLoopNestSetA, copyLoopNestSetB;
  launchOp.walk([&](ForOp loop) {
    if (checkIfCopyLoop(loop)) {
      if (isa<ForOp>(loop->getParentOp()) &&
          (loop->getParentOp()->getNumResults() > 0))
        copyLoopNestSetA.push_back(loop);
      else if (isa<IfOp>(loop->getParentOp()))
        copyLoopNestSetB.push_back(loop);
    }
  });

  if (copyLoopNestSetA.empty() || copyLoopNestSetB.empty()) {
    llvm::outs() << "I'm here\n";
    LLVM_DEBUG(llvm::dbgs() << "Warps not distributed because double buffering "
                               "has not been performed\n");
    return;
  }

  OpBuilder opb(copyLoopNestSetA[0]);
  Location loc = copyLoopNestSetA[0].getLoc();
  Value constantTwo = opb.create<ConstantIndexOp>(loc, 2);
  Value linearTidXYZ = copyLoopNestSetA[0].lowerBound();
  Value ifOpACondition, ifOpBCondition, splitPoint;
  IfOp ifOpA, ifOpB;
  Value numThreadsXYZ = copyLoopNestSetA[0].step();
  if (splitIndex > 0) {
    splitPoint = opb.create<ConstantIndexOp>(loc, splitIndex);
    ifOpACondition =
        opb.create<CmpIOp>(loc, CmpIPredicate::slt, linearTidXYZ, splitPoint);
    ifOpBCondition =
        opb.create<CmpIOp>(loc, CmpIPredicate::sge, linearTidXYZ, splitPoint);
  } else {
    splitPoint = opb.create<UnsignedDivIOp>(loc, numThreadsXYZ, constantTwo);
    ifOpACondition =
        opb.create<CmpIOp>(loc, CmpIPredicate::slt, linearTidXYZ, splitPoint);
    ifOpBCondition =
        opb.create<CmpIOp>(loc, CmpIPredicate::sge, linearTidXYZ, splitPoint);
  }
  opb.setInsertionPointAfter(copyLoopNestSetA[copyLoopNestSetA.size() - 1]);
  ifOpA = opb.create<IfOp>(loc, ifOpACondition, false);

  opb.setInsertionPoint(copyLoopNestSetB[0]->getParentOfType<IfOp>());
  // Value ifOpBCondition = opb.create<CmpIOp>(loc, CmpIPredicate::sge,
  //                                          linearTidXYZ, halfNumThreadsXYZ);

  ifOpB = opb.create<IfOp>(loc, ifOpBCondition, false);

  // Moving copy loops inside if ops.
  for (ForOp loop : copyLoopNestSetA)
    moveBody<ForOp>(loop, ifOpA);

  IfOp copyLoopParIfOp = copyLoopNestSetB[0]->getParentOfType<IfOp>();
  ForOp ifOpParLoop = copyLoopParIfOp->getParentOfType<ForOp>();
  opb.setInsertionPointToStart(&ifOpB.thenRegion().front());
  loc = ifOpB.getLoc();
  Value twiceOfStep = opb.create<MulIOp>(loc, ifOpParLoop.step(), constantTwo);
  Value copyLoopParIfOpBound =
      opb.create<SubIOp>(loc, ifOpParLoop.upperBound(), twiceOfStep);
  Value constantZero = opb.create<ConstantIndexOp>(loc, 0);
  Value copyLoopParentIfOpCond = opb.create<CmpIOp>(
      loc, CmpIPredicate::sge, copyLoopParIfOpBound, constantZero);
  copyLoopParIfOp.setOperand(copyLoopParentIfOpCond);

  moveBody<IfOp>(copyLoopParIfOp, ifOpB);
  opb.setInsertionPoint(copyLoopParIfOp);
  Value copyLoopSetTwoLB = opb.create<SubIOp>(loc, linearTidXYZ, splitPoint);
  Value newStep = opb.create<SubIOp>(loc, numThreadsXYZ, splitPoint);
  for (ForOp copyLoop : copyLoopNestSetB) {
    copyLoop.setLowerBound(copyLoopSetTwoLB);
    copyLoop.setStep(newStep);
  }
  for (ForOp copyLoop : copyLoopNestSetA)
    copyLoop.setStep(splitPoint);
}

void TestGpuMatmulDistributeWarpsAmongLoops::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  moduleOp.walk([&](gpu::LaunchOp launchOp) {
    distributeWarpsAmongCopyLoops(launchOp, splitIndex);
  });
}

namespace mlir {
void registerTestGpuMatmulDistributeWarpsAmongLoopsPass() {
  PassRegistration<TestGpuMatmulDistributeWarpsAmongLoops>(
      "test-gpu-matmul-distribute-warps-among-loops",
      "Distribute warps among loops for matmul.");
}
} // namespace mlir
