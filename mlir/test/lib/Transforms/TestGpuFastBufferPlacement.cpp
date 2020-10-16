//= TestAffineLoopParametricTiling.cpp -- Parametric Affine loop tiling pass =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to parametrically tile perfectly nested affine
// loops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/Support/Debug.h"
using namespace mlir;

#define DEBUG_TYPE "test-affine-parametric-tile"

namespace {

struct TestGPUFastBufferPlacement
    : public PassWrapper<TestGPUFastBufferPlacement, FunctionPass> {
  void runOnFunction() override;
};
} // end anonymous namespace

LogicalResult createAndPlaceFastBuffers(AffineForOp rootForOp,
                                        OpBuilder opBuilder) {
  // Get the outermost two loops as parallel loops mapped to GPU thread blocks,
  // allocation must happen right under these loops. The root forop and it's
  // immediate child are the candidates for now.
  // The loops are perfectly nested in this phase of the IR.
  SmallVector<AffineForOp, 6> loopNest;
  getPerfectlyNestedLoops(loopNest, rootForOp);

  Value outputMemRef, lhsMemRef, rhsMemRef;
  // Identify the LHS, RHS, and output memrefs.
  rootForOp.walk(
      [&](AffineStoreOp storeOp) { outputMemRef = storeOp.getMemRef(); });
  rootForOp.walk([&](AffineLoadOp loadOp) {
    if (outputMemRef == loadOp.getMemRef())
      return;
    rhsMemRef = loadOp.getMemRef();
  });
  rootForOp.walk([&](AffineLoadOp loadOp) {
    if (rhsMemRef == loadOp.getMemRef() || outputMemRef == loadOp.getMemRef())
      return;
    lhsMemRef = loadOp.getMemRef();
  });

  // Allocate fast memory buffers.
  AffineCopyOptions copyOptions = {/*generateDma=*/false,
                                   /*slowMemorySpace=*/0,
                                   /*fastMemorySpace=*/3,
                                   /*tagMemorySpace=*/0,
                                   /*fastMemCapacityBytes=*/UINT_MAX,
                                   AffineMap(),
                                   loopNest[2].getBody()->begin(),
                                   std::prev(loopNest[2].getBody()->end()),
                                   loopNest[2].getBody(),
                                   loopNest[1].getBody(),
                                   true,
                                   false};
  DenseSet<Operation *> copyNests;
  affineDataCopyGenerate(loopNest[2].getBody()->begin(),
                         std::prev(loopNest[2].getBody()->end()), copyOptions,
                         lhsMemRef, copyNests);

  affineDataCopyGenerate(loopNest[3].getBody()->begin(),
                         std::prev(loopNest[3].getBody()->end()), copyOptions,
                         rhsMemRef, copyNests);

  for (Operation *copyNest : copyNests) {
    copyNest->setAttr("isCopyLoop", opBuilder.getBoolAttr(true));
  }

  // if (kC) {
  //  // RHS matrix, pack into L3 tile if the kC loop exists.
  //  copyOptions.fastBufferLayout = AffineMap();
  //  affineDataCopyGenerate(kC.getBody()->begin(),
  //                         std::prev(kC.getBody()->end()), copyOptions,
  //                         rhsMemRef, copyNests, &fastBuf);
  //  rhsL3Buf = fastBuf[0];
  //} else {
  //  rhsL3Buf = rhsMemRef;
  //}

  //// For the RHS matrix (pack into L1).
  // copyOptions.fastBufferLayout = AffineMap();
  // copyOptions.fastMemCapacityBytes = 256 * 1024UL;
  // affineDataCopyGenerate(jR.getBody()->begin(),
  //                       std::prev(jR.getBody()->end()), copyOptions,
  //                       /*filterMemRef=*/rhsL3Buf, copyNests, &fastBuf);
  // rhsL1Buf = fastBuf[0];

  //// Set alignment to 256-bit boundaries for LHS and RHS buffers.
  //// FIXME: you don't need to set alignment if these are already vector
  //// memrefs.
  // cast<AllocOp>(lhsBuf.getDefiningOp())
  //    .setAttr(AllocOp::getAlignmentAttrName(),
  //             opBuilder.getI64IntegerAttr(32));
  // The rhsL3buf could sometimes just be the original memref / func arg.
  // if (auto rhsAllocOp = rhsL3Buf.getDefiningOp())
  //  rhsAllocOp->setAttr(AllocOp::getAlignmentAttrName(),
  //                      opBuilder.getI64IntegerAttr(32));
  // cast<AllocOp>(rhsL1Buf.getDefiningOp())
  //    .setAttr(AllocOp::getAlignmentAttrName(),
  //             opBuilder.getI64IntegerAttr(32));

  return success();
}

void runOnBlock(Block &block) {
  for (Operation &op : block) {
    if (AffineForOp forOp = dyn_cast<AffineForOp>(op)) {
      if (!isa<AffineForOp>(forOp.getParentOp())) {
        OpBuilder opBuilder(forOp);
        createAndPlaceFastBuffers(forOp, opBuilder);
      }
    }
  }
}

void TestGPUFastBufferPlacement::runOnFunction() {
  FuncOp funcOp = getFunction();

  for (Block &block : funcOp) {
    runOnBlock(block);
  }
}

namespace mlir {
void registerTestGpuFastBufferPlacementPass() {
  PassRegistration<TestGPUFastBufferPlacement>(
      "test-gpu-fast-buffer-placement",
      "Place fast memory(SMEM) buffers right inside kernel");
}
} // namespace mlir
