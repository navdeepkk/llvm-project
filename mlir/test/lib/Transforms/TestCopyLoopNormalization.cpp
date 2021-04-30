//===-------- TestLoopUnrolling.cpp --- loop unrolling test pass ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to unroll loops by a specified unroll factor.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

class TestCopyLoopNormalization
    : public PassWrapper<TestCopyLoopNormalization, FunctionPass> {
public:
  TestCopyLoopNormalization() = default;
  TestCopyLoopNormalization(const TestCopyLoopNormalization &) {}
  explicit TestCopyLoopNormalization(std::string copyLoopStrAttr) {
    copyLoopAttr = copyLoopStrAttr;
  }

  void runOnFunction() override {
    FuncOp func = getFunction();
    SmallVector<AffineForOp> loops;
    func.walk([&](AffineForOp forOp) {
      BoolAttr isCopyLoop = forOp->getAttrOfType<BoolAttr>(copyLoopAttr);
      if (isCopyLoop && isCopyLoop.getValue() == true)
        loops.push_back(forOp);
    });
    for (auto loop : loops)
      normalizeAffineFor(loop);
  }
  Option<std::string> copyLoopAttr{
      *this, "copy-loop-attr", llvm::cl::desc("copy loop identifier attribute"),
      llvm::cl::init("isCopyLoopNest")};
};
} // namespace

namespace mlir {
namespace test {
void registerTestCopyLoopNormalization() {
  PassRegistration<TestCopyLoopNormalization>("test-normalize-copy-loops",
                                              "Unroll copy loops");
}
} // namespace test
} // namespace mlir
