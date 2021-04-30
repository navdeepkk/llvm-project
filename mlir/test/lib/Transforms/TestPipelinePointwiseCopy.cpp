#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace {

/// The existing -pipeline-data-transfer pass supports pipelining for DMA ops
/// only. This pass adds support for pipelining of pointwise copy loops using
/// double buffering i.e. in each iteration the data is copied to one memory
/// location and for computation the data is loaded from alternate memory
/// location. In this way data copy and computation both happens in parallel.
///
/// The pass uses `clIdentifyCopyLoopNestAttribute` to identify that the given
/// loop nest is a copy loop nest which copies data from global memory to shared
/// memory and pipelining is done in such a way that one round of execution of
/// this loop nest is done in advance so that during computation, the
/// computation loop nest will use the data copied in previous iteration of copy
/// loop nest, meanwhile the copy loop nest will copy data for next iteration of
/// compute loop nest.
///
/// `clIdentifyComputeLoopNestAttribute` is used to identify that the given loop
/// nest is a computation loop nest which performs computation and pipelining is
/// done in such a way that this loop nest starts execution after one iteration
/// of copy loop nest is executed.
///
/// `clMemorySpace` specifies which memory space buffers to be used for double
/// buffering.
struct TestPipelinePointwiseCopy
    : public PassWrapper<TestPipelinePointwiseCopy, FunctionPass> {
  TestPipelinePointwiseCopy() = default;
  TestPipelinePointwiseCopy(const TestPipelinePointwiseCopy &pass) {}

  void runOnFunction() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
  }

private:
  Option<std::string> clIdentifyCopyLoopNestAttribute{
      *this, "copy-loop-nest-attr",
      llvm::cl::desc("Used to identify loop nest(s) which copy data"),
      llvm::cl::init("isCopyLoopNest")};
  Option<std::string> clIdentifyComputeLoopNestAttribute{
      *this, "compute-loop-nest-attr",
      llvm::cl::desc(
          "Used to identify loop nest(s) which performs computation"),
      llvm::cl::init("isComputeLoopNest")};
};

} // end of namespace

// Runs on a function 'func' and performs the pipelining of pointwise copy
// loops using double buffering.
void TestPipelinePointwiseCopy::runOnFunction() {
  FuncOp funcOp = getFunction();
  // pipelinePointwiseCopy(func, clIdentifyCopyLoopNestAttribute,
  //                      clIdentifyComputeLoopNestAttribute);
  OpBuilder b(funcOp.getContext());
  bool insertBarrier = false;
  Operation *toSplit = nullptr;
  funcOp.walk([&](AffineForOp forOp) {
    BoolAttr attr =
        forOp->getAttrOfType<BoolAttr>(clIdentifyComputeLoopNestAttribute);
    if (attr && attr.getValue() == true) {
      splitLoop(forOp, clIdentifyCopyLoopNestAttribute,
                clIdentifyComputeLoopNestAttribute);
      insertBarrier = true;
      toSplit = forOp.getOperation();
    }
  });
  // b.setInsertionPointAfter(toSplit);
  // b.create<gpu::BarrierOp>(funcOp.getLoc());
}

namespace mlir {
void registerTestPipelinePointwiseCopyPass() {
  PassRegistration<TestPipelinePointwiseCopy>(
      "test-pipeline-pointwise-copy",
      "Tests pipelining with double buffering supported");
}
} // namespace mlir
