//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

// Defined in the test directory, no public header.
namespace mlir {
void registerConvertToTargetEnvPass();
void registerPassManagerTestPass();
void registerPrintOpAvailabilityPass();
void registerShapeFunctionTestPasses();
void registerSideEffectTestPasses();
void registerSliceAnalysisTestPass();
void registerSymbolTestPasses();
void registerTestAffineDataCopyPass();
void registerTestAffineLoopUnswitchingPass();
void registerTestAllReduceLoweringPass();
void registerTestFunc();
void registerTestGpuMarkGlobalAsWorkgroupMemoryPass();
void registerTestGPUMatmulBarrierPlacement();
void registerTestGpuMatmulFastBufferPlacementPass();
void registerTestGpuMatmulDistributeWarpsAmongLoopsPass();
void registerTestGpuMemoryPromotionPass();
void registerTestGpuMatmulParallelLoopMappingPass();
void registerTestCollapseAffineParallelPass();
void registerTestConvertMatmulParallelLoopsToGPUPass();
void registerTestLoopPermutationPass();
void registerTestMatchers();
void registerTestMarkParallelLoops();
void registerTestPipelinePointwiseCopyPass();
void registerTestPrintDefUsePass();
void registerTestPrintNestingPass();
void registerTestReducer();
void registerTestSpirvEntryPointABIPass();
void registerTestSpirvGLSLCanonicalizationPass();
void registerTestSpirvModuleCombinerPass();
void registerTestTraitsPass();
void registerTestVectorizeGPUMatmulCopyLoops();
void registerTosaTestQuantUtilAPIPass();
void registerVectorizerTestPass();
void registerTestUnrollAndDelayCopiesPass();

namespace test {
void registerConvertCallOpPass();
void registerInliner();
void registerMemRefBoundCheck();
void registerPatternsTestPass();
void registerSimpleParametricTilingPass();
void registerTestAffineLoopParametricTilingPass();
void registerTestAliasAnalysisPass();
void registerTestCallGraphPass();
void registerTestConstantFold();
void registerTestConvVectorization();
void registerTestConvertGPUKernelToCubinPass();
void registerTestConvertGPUKernelToHsacoPass();
void registerTestDecomposeCallGraphTypes();
void registerTestDialect(DialectRegistry &);
void registerTestDominancePass();
void registerTestDynamicPipelinePass();
void registerTestExpandTanhPass();
void registerTestGpuParallelLoopMappingPass();
void registerTestInterfaces();
void registerTestLinalgCodegenStrategy();
void registerTestLinalgFusionTransforms();
void registerTestLinalgTensorFusionTransforms();
void registerTestLinalgGreedyFusion();
void registerTestLinalgHoisting();
void registerTestLinalgTileAndFuseSequencePass();
void registerTestLinalgTransforms();
void registerTestLivenessPass();
void registerTestLoopFusion();
void registerTestLoopMappingPass();
void registerTestLoopUnrollingPass();
void registerTestSpecificLoopUnrollingPass();
void registerTestSCFSpecificLoopUnrollingPass();
void registerTestCopyLoopNormalization();
void registerTestMathPolynomialApproximationPass();
void registerTestMemRefDependenceCheck();
void registerTestMemRefStrideCalculation();
void registerTestNumberOfBlockExecutionsPass();
void registerTestNumberOfOperationExecutionsPass();
void registerTestOpaqueLoc();
void registerTestPDLByteCodePass();
void registerTestPreparationPassWithAllowedMemrefResults();
void registerTestRecursiveTypesPass();
void registerTestSCFUtilsPass();
void registerTestSparsification();
void registerTestSpecializeAffineForWMMAPass();
void registerTestVectorConversions();
} // namespace test
} // namespace mlir

#ifdef MLIR_INCLUDE_TESTS
void registerTestPasses() {
  registerConvertToTargetEnvPass();
  registerPassManagerTestPass();
  registerPrintOpAvailabilityPass();
  registerShapeFunctionTestPasses();
  registerSideEffectTestPasses();
  registerSliceAnalysisTestPass();
  registerSymbolTestPasses();
  registerTestAffineDataCopyPass();
  registerTestAffineLoopUnswitchingPass();
  registerTestAllReduceLoweringPass();
  registerTestFunc();
  registerTestGpuMemoryPromotionPass();
  registerTestGpuMarkGlobalAsWorkgroupMemoryPass();
  registerTestGpuMatmulDistributeWarpsAmongLoopsPass();
  registerTestGpuMatmulParallelLoopMappingPass();
  registerTestGPUMatmulBarrierPlacement();
  registerTestCollapseAffineParallelPass();
  registerTestConvertMatmulParallelLoopsToGPUPass();
  registerTestLoopPermutationPass();
  registerTestGpuMatmulFastBufferPlacementPass();
  registerTestMatchers();
  registerTestMarkParallelLoops();
  registerTestPipelinePointwiseCopyPass();
  registerTestPrintDefUsePass();
  registerTestPrintNestingPass();
  registerTestReducer();
  registerTestSpirvEntryPointABIPass();
  registerTestSpirvGLSLCanonicalizationPass();
  registerTestSpirvModuleCombinerPass();
  registerTestVectorizeGPUMatmulCopyLoops();
  registerTestTraitsPass();
  registerVectorizerTestPass();
  registerTosaTestQuantUtilAPIPass();
  registerTestUnrollAndDelayCopiesPass();

  test::registerConvertCallOpPass();
  test::registerInliner();
  test::registerMemRefBoundCheck();
  test::registerPatternsTestPass();
  test::registerSimpleParametricTilingPass();
  test::registerTestAffineLoopParametricTilingPass();
  test::registerTestAliasAnalysisPass();
  test::registerTestCallGraphPass();
  test::registerTestConstantFold();
#if MLIR_CUDA_CONVERSIONS_ENABLED
  test::registerTestConvertGPUKernelToCubinPass();
#endif
#if MLIR_ROCM_CONVERSIONS_ENABLED
  test::registerTestConvertGPUKernelToHsacoPass();
#endif
  test::registerTestConvVectorization();
  test::registerTestDecomposeCallGraphTypes();
  test::registerTestDominancePass();
  test::registerTestDynamicPipelinePass();
  test::registerTestExpandTanhPass();
  test::registerTestGpuParallelLoopMappingPass();
  test::registerTestInterfaces();
  test::registerTestLinalgCodegenStrategy();
  test::registerTestLinalgFusionTransforms();
  test::registerTestLinalgTensorFusionTransforms();
  test::registerTestLinalgGreedyFusion();
  test::registerTestLinalgHoisting();
  test::registerTestLinalgTileAndFuseSequencePass();
  test::registerTestLinalgTransforms();
  test::registerTestLivenessPass();
  test::registerTestLoopFusion();
  test::registerTestLoopMappingPass();
  test::registerTestLoopUnrollingPass();
  test::registerTestSpecificLoopUnrollingPass();
  test::registerTestSCFSpecificLoopUnrollingPass();
  test::registerTestCopyLoopNormalization();
  test::registerTestMathPolynomialApproximationPass();
  test::registerTestMemRefDependenceCheck();
  test::registerTestMemRefStrideCalculation();
  test::registerTestNumberOfBlockExecutionsPass();
  test::registerTestNumberOfOperationExecutionsPass();
  test::registerTestOpaqueLoc();
  test::registerTestPDLByteCodePass();
  test::registerTestRecursiveTypesPass();
  test::registerTestSCFUtilsPass();
  test::registerTestSparsification();
  test::registerTestSpecializeAffineForWMMAPass();
  test::registerTestVectorConversions();
}
#endif

int main(int argc, char **argv) {
  registerAllPasses();
#ifdef MLIR_INCLUDE_TESTS
  registerTestPasses();
#endif
  DialectRegistry registry;
  registerAllDialects(registry);
#ifdef MLIR_INCLUDE_TESTS
  test::registerTestDialect(registry);
#endif
  return failed(MlirOptMain(argc, argv, "MLIR modular optimizer driver\n",
                            registry,
                            /*preloadDialectsInContext=*/false));
}
