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
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

class TestSCFSpecificLoopUnrollingPass
    : public PassWrapper<TestSCFSpecificLoopUnrollingPass, FunctionPass> {
public:
  TestSCFSpecificLoopUnrollingPass() = default;
  TestSCFSpecificLoopUnrollingPass(const TestSCFSpecificLoopUnrollingPass &) {}
  explicit TestSCFSpecificLoopUnrollingPass(std::string unrollStrAttr) {
    unrollAttr = unrollStrAttr;
  }

  void runOnFunction() override {
    FuncOp func = getFunction();
    SmallVector<scf::ForOp, 4> loops;
    func.walk([&](scf::ForOp forOp) {
      if (forOp->getAttrOfType<BoolAttr>(unrollAttr))
        loops.push_back(forOp);
    });
    for (auto loop : loops)
      (void)loopUnrollByFactor(loop, 2);
  }
  Option<std::string> unrollAttr{*this, "unroll-attr",
                                 llvm::cl::desc("Loop to Unroll"),
                                 llvm::cl::init("isCopyLoopNest")};
};
} // namespace

namespace mlir {
namespace test {
void registerTestSCFSpecificLoopUnrollingPass() {
  PassRegistration<TestSCFSpecificLoopUnrollingPass>(
      "test-unroll-scf-specific-loops", "Tests loop unrolling transformation");
}
} // namespace test
} // namespace mlir
