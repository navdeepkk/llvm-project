//===------ TestConvertToGpu.cpp ----------- convert loops to gpu --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test the conversion of loops to gpu.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/ParallelLoopMapper.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::scf;

namespace {
class TestConvertToGpuPass
    : public PassWrapper<TestConvertToGpuPass, OperationPass<>> {
public:
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect>();
    registry.insert<gpu::GPUDialect>();
  }
  TestConvertToGpuPass(){};
  TestConvertToGpuPass(const TestConvertToGpuPass &) {}
  explicit TestConvertToGpuPass(ArrayRef<int64_t> tbSizes) {
    tbDimsRef = tbSizes;
  }

  void filltbDims();

  ListOption<int64_t> tbDimsRef{
      *this, "launch-params", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc("List of function name to apply the pipeline to")};

  SmallVector<int64_t, 3> tbDims;
};
} // end namespace

namespace {
enum MappingLevel { threads, warps, threadBlocks };

static bool isMappedToProcessor(gpu::Processor processor) {
  return processor != gpu::Processor::Sequential;
}

static unsigned getLaunchOpArgumentNum(gpu::Processor processor) {
  switch (processor) {
  case gpu::Processor::BlockX:
    return 0;
  case gpu::Processor::BlockY:
    return 1;
  case gpu::Processor::BlockZ:
    return 2;
  case gpu::Processor::ThreadX:
    return 3;
  case gpu::Processor::ThreadY:
    return 4;
  case gpu::Processor::ThreadZ:
    return 5;
  default:;
  }
  llvm_unreachable(
      "invalid processor type while retrieving launch op argument number");
}

bool checkIfCopyLoop(ParallelOp parallelOp) {
  BoolAttr copyLoop = parallelOp.getAttrOfType<BoolAttr>("isCopyLoop");
  if (copyLoop) {
    if (copyLoop.getValue())
      return true;
  }

  return false;
}

struct LoopsToGpuLowering : public OpRewritePattern<ParallelOp> {
  using OpRewritePattern<ParallelOp>::OpRewritePattern;
  explicit LoopsToGpuLowering(MLIRContext *context, ArrayRef<int64_t> tbSizes)
      : OpRewritePattern<ParallelOp>(context), tbDims(tbSizes) {}

  LogicalResult matchAndRewrite(ParallelOp parallelOp,
                                PatternRewriter &rewriter) const override;

private:
  ArrayRef<int64_t> tbDims;
};
} // namespace

bool insertLaunchParams(ParallelOp parallelOp, ArrayRef<int64_t> tbDims,
                        PatternRewriter &rewriter, Location &topLoc,
                        SmallVectorImpl<Value> &tbDimValues,
                        SmallVectorImpl<Value> &gridDimValues) {

  // Each loop of the parallel op will be mapped to one of the grid dimensions.
  // If the number of loops in th parallel op is greater than 3 then fail.
  // TODO: Handle cases where the number of loops is greater than 3.
  if (parallelOp.getNumLoops() > 3)
    return false;

  // Create ops for dimesnions of thread block.
  for (int64_t param : tbDims) {
    tbDimValues.push_back(rewriter.create<ConstantIndexOp>(topLoc, param));
  }

  // Create ops for dimesnions of grid. The grid dimesnions are dependent on the
  // dimensions of the thread block and the number of iteratoins of the loop
  // being mapped. Currently making a simple assumption that one thread block
  // will calculate the	loopStep amount of iterations. Hence the number of
  // thread blocks to be launched will be (loopUB - loopLB) / loopStep for a
  // particular loop.
  for (auto loop : llvm::zip(parallelOp.lowerBound(), parallelOp.upperBound(),
                             parallelOp.step())) {
    Value lowerBound, upperBound, step;
    std::tie(lowerBound, upperBound, step) = loop;
    // Check if the defining ops for upper/lower bound and step are constant.
    // If not the return false.
    if (isa<ConstantIndexOp>(lowerBound.getDefiningOp()) &&
        isa<ConstantIndexOp>(upperBound.getDefiningOp()) &&
        isa<ConstantIndexOp>(step.getDefiningOp())) {
      Value diff = rewriter.create<SubIOp>(topLoc, upperBound, lowerBound);
      gridDimValues.push_back(
          rewriter.create<SignedDivIOp>(topLoc, diff, step));
    } else
      return false;
  }

  return true;
}

LogicalResult convertIfOp(gpu::LaunchOp launchOp, IfOp ifOp,
                          MappingLevel &mappingLevel,
                          BlockAndValueMapping &cloningMap,
                          SmallVectorImpl<Operation *> &worklist,
                          PatternRewriter &rewriter) {
  // The IfOp haves both `ifThen` part and `else` part. Both of them have to
  // be copied over.
  bool hasElseRegion = ifOp.elseRegion().empty() ? false : true;

  Location loc = ifOp.getLoc();
  auto clonedIfOp = rewriter.create<scf::IfOp>(
      loc, cloningMap.lookupOrDefault(ifOp.condition()), hasElseRegion);

  // First insert the sentinal values which marks the end of the ifOp scope.
  worklist.push_back(launchOp.getOperation());

  // Now insert the body of the else part into the worklist.
  if (hasElseRegion) {
    Block *body = &ifOp.elseRegion().front();
    worklist.reserve(worklist.size() + body->getOperations().size());
    for (Operation &op : llvm::reverse(body->without_terminator())) {
      worklist.push_back(&op);
    }
    // The sentinal for the end of else region is inserted now. The newly
    // created IfOp is used as the sentinal value. To prevent this IfOp from
    // being processed again we while processing if the IfOp has a gpu.launch op
    // as a parent. If yes then this is our sentinal value.
    worklist.push_back(clonedIfOp.getOperation());
  }

  // Now insert the body of the then part into the worklist.
  rewriter.setInsertionPointToStart(&clonedIfOp.thenRegion().front());
  Block *body = &ifOp.thenRegion().front();
  worklist.reserve(worklist.size() + body->getOperations().size());
  for (Operation &op : llvm::reverse(body->without_terminator())) {
    worklist.push_back(&op);
  }

  return success();
}
LogicalResult convertForLoop(gpu::LaunchOp launchOp, ForOp forOp,
                             MappingLevel &mappingLevel,
                             BlockAndValueMapping &cloningMap,
                             SmallVectorImpl<Operation *> &worklist,
                             PatternRewriter &rewriter) {
  Location loc = forOp.getLoc();
  auto loopOp = rewriter.create<scf::ForOp>(
      loc, cloningMap.lookupOrDefault(forOp.lowerBound()),
      cloningMap.lookupOrDefault(forOp.upperBound()),
      cloningMap.lookupOrDefault(forOp.step()));
  Value newIndex = loopOp.getInductionVar();
  rewriter.setInsertionPointToStart(loopOp.getBody());
  // Put a sentinel into the worklist so we know when to pop out of the loop
  // body again. We use the launchOp here, as that cannot be part of the bodies
  // instruction.
  worklist.push_back(launchOp.getOperation());
  cloningMap.map(forOp.getInductionVar(), newIndex);

  Block *body = forOp.getBody();
  worklist.reserve(worklist.size() + body->getOperations().size());
  for (Operation &op : llvm::reverse(body->without_terminator())) {
    worklist.push_back(&op);
  }

  return success();
}

LogicalResult convertParallelLoop(gpu::LaunchOp launchOp, ParallelOp parallelOp,
                                  MappingLevel &mappingLevel,
                                  BlockAndValueMapping &cloningMap,
                                  SmallVectorImpl<Operation *> &worklist,
                                  PatternRewriter &rewriter) {
  // Since the transformation assumes the loops have been normalized, mapping is
  // as simple as just replacing the loop ivs with the appropriate hardware ids.
  bool isCopyLoop = checkIfCopyLoop(parallelOp);
  Location loc = parallelOp.getLoc();
  if (isCopyLoop) {
    // Copy loops are handeled specially. A copy loop is assumed to be 1-d and
    // is distributed among the threads in a lineat fashion so as to enable
    // global memory coallescing.
    // TODO: Enable further optimizations such as prevention of shared memory
    // bank conflicts while loading the operands. Insert ops for calculating
    // the linear thread Id.
    for (auto loop :
         llvm::zip(parallelOp.getInductionVars(), parallelOp.lowerBound(),
                   parallelOp.upperBound(), parallelOp.step())) {
      Value iv, lowerBound, upperBound, step;
      std::tie(iv, lowerBound, upperBound, step) = loop;
      Value xdimYdim = rewriter.create<MulIOp>(loc, launchOp.blockSizeX(),
                                               launchOp.blockSizeY());
      Value zIdXdimYdim =
          rewriter.create<MulIOp>(loc, launchOp.getThreadIds().z, xdimYdim);
      Value yIdXdim = rewriter.create<MulIOp>(loc, launchOp.getThreadIds().y,
                                              launchOp.blockSizeX());
      Value linearTidyz = rewriter.create<AddIOp>(loc, zIdXdimYdim, yIdXdim);
      Value linearTidxyz =
          rewriter.create<AddIOp>(loc, linearTidyz, launchOp.getThreadIds().x);
      Value numThreadsxy = rewriter.create<MulIOp>(loc, launchOp.blockSizeX(),
                                                   launchOp.blockSizeY());
      Value numThreadsxyz =
          rewriter.create<MulIOp>(loc, numThreadsxy, launchOp.blockSizeZ());
      auto loopOp = rewriter.create<scf::ForOp>(
          loc, linearTidxyz, cloningMap.lookupOrDefault(upperBound),
          numThreadsxyz);
      Value newIndex = loopOp.getInductionVar();
      rewriter.setInsertionPointToStart(loopOp.getBody());
      // Put a sentinel into the worklist so we know when to pop out of the
      // loop body again. We use the launchOp here, as that cannot be part of
      // the bodies instruction.
      worklist.push_back(launchOp.getOperation());
      cloningMap.map(iv, newIndex);
    }
  } else {
    ArrayAttr mapping =
        parallelOp.getAttrOfType<ArrayAttr>(gpu::getMappingAttrName());

    for (auto loop : llvm::zip(mapping, parallelOp.getInductionVars(),
                               parallelOp.lowerBound(), parallelOp.upperBound(),
                               parallelOp.step())) {
      Value iv, lowerBound, upperBound, step;
      Attribute mappingAttribute;
      std::tie(mappingAttribute, iv, lowerBound, upperBound, step) = loop;
      auto annotation =
          mappingAttribute.dyn_cast<gpu::ParallelLoopDimMapping>();
      if (!annotation)
        return parallelOp.emitOpError()
               << "expected mapping attribute for lowering to GPU";
      gpu::Processor processor = gpu::getProcessor(annotation);
      if (isMappedToProcessor(processor)) {
        if (processor < gpu::Processor::ThreadX) {
          // Use the corresponding thread/grid index as replacement for the loop
          // iv.
          Value operand =
              launchOp.body().getArgument(getLaunchOpArgumentNum(processor));
          cloningMap.map(iv, operand);
        } else {
          // The parallel op is mapped to threads. For now distribute this
          // cyclically among the threads in a thread block. In a cyclic
          // distribution the lower bound of the loop is equal to the thread id
          // in the corresponding dimension. The upper bound need not be
          // changed. The step is equal to the thread block size in the
          // corresponding dimension.
          // TODO: Intorduce the type of distribution as an attribute and
          // distribute the loop accordingly.
          auto loopOp = rewriter.create<scf::ForOp>(
              loc,
              launchOp.body().getArgument(getLaunchOpArgumentNum(processor)),
              cloningMap.lookupOrDefault(upperBound),
              cloningMap.lookupOrDefault(
                  launchOp.getOperand(getLaunchOpArgumentNum(processor))));
          Value newIndex = loopOp.getInductionVar();
          rewriter.setInsertionPointToStart(loopOp.getBody());
          // Put a sentinel into the worklist so we know when to pop out of the
          // loop body again. We use the launchOp here, as that cannot be part
          // of the bodies instruction.
          worklist.push_back(launchOp.getOperation());
          cloningMap.map(iv, newIndex);
        }
      }
    }
  }

  Block *body = parallelOp.getBody();
  worklist.reserve(worklist.size() + body->getOperations().size());
  for (Operation &op : llvm::reverse(body->without_terminator())) {
    worklist.push_back(&op);
  }

  return success();
}

LogicalResult
LoopsToGpuLowering::matchAndRewrite(ParallelOp parallelOp,
                                    PatternRewriter &rewriter) const {
  Value constantOne = rewriter.create<ConstantIndexOp>(parallelOp.getLoc(), 1);
  MappingLevel mappingLevel = threadBlocks;
  Location topLoc = parallelOp.getLoc();
  SmallVector<Value, 3> tbDimValues, gridDimValues;
  if (!insertLaunchParams(parallelOp, tbDims, rewriter, topLoc, tbDimValues,
                          gridDimValues))
    return failure();
  gridDimValues.insert(gridDimValues.end(), 3 - gridDimValues.size(),
                       constantOne);

  gpu::LaunchOp launchOp = rewriter.create<gpu::LaunchOp>(
      parallelOp.getLoc(), gridDimValues[0], gridDimValues[1], gridDimValues[2],
      tbDimValues[0], tbDimValues[1], tbDimValues[2]);
  rewriter.setInsertionPointToEnd(&launchOp.body().front());
  rewriter.create<gpu::TerminatorOp>(topLoc);
  rewriter.setInsertionPointToStart(&launchOp.body().front());

  BlockAndValueMapping cloningMap;
  SmallVector<Operation *, 16> worklist;
  if (failed(convertParallelLoop(launchOp, parallelOp, mappingLevel, cloningMap,
                                 worklist, rewriter)))
    return failure();

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (auto nestedParallel = dyn_cast<ParallelOp>(op)) {
      if (failed(convertParallelLoop(launchOp, nestedParallel, mappingLevel,
                                     cloningMap, worklist, rewriter)))
        return failure();
    } else if (op == launchOp.getOperation()) {
      auto parent = rewriter.getInsertionPoint()->getParentOp();
      rewriter.setInsertionPointAfter(parent);
    } else if (auto nestedFor = dyn_cast<ForOp>(op)) {
      if (failed(convertForLoop(launchOp, nestedFor, mappingLevel, cloningMap,
                                worklist, rewriter)))
        return failure();
    } else if (auto nestedIf = dyn_cast<IfOp>(op)) {
      if (nestedIf.getParentOfType<gpu::LaunchOp>() == launchOp) {
        // This is a sentinal op. Set the rewriter to the then part of the if
        // op.
        if (IfOp parent =
                dyn_cast<IfOp>(rewriter.getInsertionPoint()->getParentOp()))
          rewriter.setInsertionPointToStart(&parent.elseRegion().front());
      } else {
        if (failed(convertIfOp(launchOp, nestedIf, mappingLevel, cloningMap,
                               worklist, rewriter)))
          return failure();
      }
    } else {
      Operation *clone = rewriter.clone(*op, cloningMap);
      cloningMap.map(op->getResults(), clone->getResults());
    }
  }

  rewriter.eraseOp(parallelOp);
  return success();
}

void populateConvertToGPUPatterns(OwningRewritePatternList &patterns,
                                  MLIRContext *ctx, ArrayRef<int64_t> tbDims) {
  patterns.insert<LoopsToGpuLowering>(ctx, tbDims);
}

void TestConvertToGpuPass::filltbDims() {
  for (int i = 0, e = tbDimsRef.size(); i < 3; ++i) {
    if (i < e)
      tbDims.push_back(tbDimsRef[i]);
    else
      tbDims.push_back(1);
  }
}

void TestConvertToGpuPass::runOnOperation() {
  filltbDims();
  OwningRewritePatternList patterns;
  populateConvertToGPUPatterns(patterns, &getContext(), tbDims);
  ConversionTarget target(getContext());
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<AffineDialect>();
  target.addLegalDialect<gpu::GPUDialect>();
  target.addLegalDialect<scf::SCFDialect>();
  target.addIllegalOp<scf::ParallelOp>();
  if (failed(applyPartialConversion(getOperation(), target, patterns)))
    signalPassFailure();
}

namespace mlir {
void registerTestConvertToGpuPass() {
  PassRegistration<TestConvertToGpuPass>(
      "test-convert-to-gpu", "Tests the mapping of parallel loops to GPU ops.");
}
} // namespace mlir
