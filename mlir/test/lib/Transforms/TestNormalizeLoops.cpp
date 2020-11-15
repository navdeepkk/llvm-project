//===- ParallelLoopCollapsing.cpp - Pass collapsing parallel loop indices -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "loop-normalization"

using namespace mlir;

namespace {
struct NormalizeLoops 
  : public PassWrapper<NormalizeLoops, FunctionPass> {
    void runOnFunction() override;
  };

  void NormalizeLoops::runOnFunction() {
    FuncOp func = getFunction();

    func.walk([&](Operation * op) {
	if(scf::ForOp forOp = dyn_cast<scf::ForOp>(op)){
	  BoolAttr isCopyLoop = forOp.getAttrOfType<BoolAttr>("isCopyLoop");
	  // Skip copy loops.
	  if(!isCopyLoop)
	    normalizeSingleLoop(forOp, forOp, forOp);
	}
    });
  }
} // namespace

namespace mlir {
void registerTestLoopNormalizationPass() {
  PassRegistration<NormalizeLoops>(
      "test-loop-normalization",
      "Normalize loops so that the step is 1 and lower bound is 0.");
}
}
