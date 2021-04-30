#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace {
struct TestUnrollAndDelayCopies
    : public PassWrapper<TestUnrollAndDelayCopies, FunctionPass> {
  TestUnrollAndDelayCopies() = default;
  TestUnrollAndDelayCopies(const TestUnrollAndDelayCopies &pass) {}

  void runOnFunction() override;

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

void TestUnrollAndDelayCopies::runOnFunction() {
  FuncOp funcOp = getFunction();
  // Find and unroll copy loops.
  funcOp.walk([&](scf::ForOp copyLoop) {
    BoolAttr isCopyLoop = copyLoop->getAttrOfType<BoolAttr>(
        TestUnrollAndDelayCopies::clIdentifyCopyLoopNestAttribute);
    if (isCopyLoop && isCopyLoop.getValue() == true) {
      (void)loopUnrollFull(copyLoop);
    }
  });
}

namespace mlir {
void registerTestUnrollAndDelayCopiesPass() {
  PassRegistration<TestUnrollAndDelayCopies>(
      "test-unroll-and-delay-copies",
      "Unroll copy loops completely and also delay stores in copy loops to "
      "happen after compute.");
}
} // namespace mlir
