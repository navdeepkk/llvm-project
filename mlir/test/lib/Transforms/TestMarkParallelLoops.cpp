//== -TestVectorizeGPUMatmulCopyLoops.cpp - Test copy loops vectorization - ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file performs vectorization of copy loops in matmul from slow(global) to
// fast(shared) memory.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
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

#define DEBUG_TYPE "test-vectorize-gpu-matmul-copy-loops"

namespace {
struct TestMarkParallelLoops
    : public PassWrapper<TestMarkParallelLoops, FunctionPass> {
  void runOnFunction() override;
  //
  /// Copy loop-nest string identifier.
  static const std::string isParallel;
};
} // namespace

void TestMarkParallelLoops::runOnFunction() {
  FuncOp funcOp = getFunction();

  // Check and mark the parallel loops in the IR.
  funcOp.walk([&](AffineForOp loop) {
    Optional<uint64_t> tripCount = getConstantTripCount(loop);
    // If the is single iteration we conservatively treat it as sequential.
    if (tripCount.hasValue() && tripCount.getValue() == 1) {
      // Do Nothing.
    }

    else if (isLoopParallel(loop)) {
      loop->setAttr(TestMarkParallelLoops::isParallel,
                    BoolAttr::get(loop.getContext(), true));
    }
  });
}

namespace mlir {
void registerTestMarkParallelLoops() {
  PassRegistration<TestMarkParallelLoops>(
      "test-mark-parallel-loops", "mark parallel loops in the given IR");
}
} // namespace mlir

const std::string TestMarkParallelLoops::isParallel = "isParallel";
