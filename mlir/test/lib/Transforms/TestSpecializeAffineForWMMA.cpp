//= TestSpecializeAffineForWMMA.cpp ----- Parametric Affine loop tiling pass =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Transforms/LoopUtils.h"

using namespace mlir;

#define DEBUG_TYPE "test-specialize-affine-matmul-for-wmma"

namespace {
struct TestSpecializeAffineForWMMA
    : public PassWrapper<TestSpecializeAffineForWMMA, FunctionPass> {
  void runOnFunction() override;

  unsigned kMaxTiledLoops = 9;
  unsigned kNumIntialLoops = 3;
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
          // TODO: Inset assertion if there are multiple non-copy loop
          // children of this for op.
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
void TestSpecializeAffineForWMMA::runOnFunction() {
  // Try to find out the loop structure and identify the levels of tiling done.
  // Get the root for op first.
  FuncOp funcOp = getFunction();

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
  for (auto loop : computeLoops) {
    loop.dump();
    llvm::outs() << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n";
  }

  // The expected number of loops 9 i.e., all matmul loops are tiled two times.
  // TODO: Add cases when all the loops are nto tiled.
  assert(computeLoops.size() == kMaxTiledLoops &&
         "Tensor core matmul recipe expects two-level tiling.");

  // Find the different type of loops. When mapped to GPU there may be three
  // different types of loops present. 1.) Inter ThreadBlock-tile loops, 2.)
  // Inter Warp-tile loops 3.) Intra Warp-tile loops.
  SmallVector<Value> outermostloopsIVs;
  for (auto loop = computeLoops.begin(),
            e = computeLoops.begin() + kNumIntialLoops;
       loop < e; ++loop) {
    outermostloopsIVs.push_back(loop->getInductionVar());
  }

  llvm::outs() << outermostloopsIVs.size() << "\n";
  // By inspecting the upper bounds and lower bounds of all the loops we can
  // find out which loops were actually tiled. If the lower or upper bound of a
  // loop depends on any of the loopsIVs of any of the outermost loops then that
  // loop is tiled. Check that the loops are in the desired ordered, i.e.,
  //		    Inter Thread-Block loops(i,j,k)
  //		      Inter Warp loops(ii, jj, kk)
  //			Intra Warp loops(iii, jjj, kkk)
  unsigned curMapStage = 0;
  for (auto loop = computeLoops.begin() + kNumIntialLoops,
            e = computeLoops.end();
       loop < e; ++loop) {
    // Insert lower/upper bound operands.
    SmallVector<Value> IVOperands;
    IVOperands.insert(IVOperands.end(), loop->getLowerBoundOperands().begin(),
                      loop->getLowerBoundOperands().end());
    IVOperands.insert(IVOperands.end(), loop->getUpperBoundOperands().begin(),
                      loop->getUpperBoundOperands().end());

    // The loops must be dependent from the outermost to the innermost loops.
    llvm::is_contained(IVOperands, )
  }
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
