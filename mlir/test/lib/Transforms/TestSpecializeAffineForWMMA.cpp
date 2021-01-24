//===- TestSpecializeAffineForWMMA.cpp- Test generation of GPU WMMA ops ---===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements rewriting patterns to specialize affine loops for matmul
// to use GPU WMMA Ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "test-specialize-affine-for-wmma"

using namespace mlir;

namespace {

struct TestSpecializeAffineForWMMA
    : public PassWrapper<TestSpecializeAffineForWMMA, FunctionPass> {
  void runOnFunction() override;
};

} // end anonymous namespace

void TestSpecializeAffineForWMMA::runOnFunction() {}

namespace mlir {
namespace test {
void registerTestSpecialieAffineForWMMA() {
  PassRegistration<TestSpecializeAffineForWMMA> pass(
      "test-specialize-affine-for-wmma",
      "Apply rewrites to specialize affine matmul loops to use GPU WMMA Ops.");
}
} // namespace test
} // namespace mlir
