//===------ TestConvertSCFToGPU.cpp ----------- convert loops to gpu
//--------===//
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
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::scf;

namespace {
class TestConvertSCFToGPUPass
    : public PassWrapper<TestConvertSCFToGPUPass, OperationPass<>> {
public:
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect>();
    registry.insert<gpu::GPUDialect>();
  }
  TestConvertSCFToGPUPass(){};
  TestConvertSCFToGPUPass(const TestConvertSCFToGPUPass &) {}
  explicit TestConvertSCFToGPUPass(ArrayRef<int64_t> tbSizes) {
    tbDimsRef = tbSizes;
  }

  ListOption<int64_t> tbDimsRef{
      *this, "launch-params", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc("List of thread block launch parameters")};

  // Default warp size is 32.
  Option<int64_t> warpSizeRef{
      *this, "warp-size", llvm::cl::desc("Size of Warp"), llvm::cl::init(32)};

  SmallVector<int64_t, 3> tbDims;

  void filltbDims();
};
} // end namespace

namespace {

static int64_t warpSize;
Value linearTidXYZ, numThreadsXYZ, linearWarpId, mTile, nTile, numWarps,
    warpMtile, warpNtile;
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
  case gpu::Processor::WarpX:
    return 3;
  case gpu::Processor::WarpY:
    return 4;
  case gpu::Processor::WarpZ:
    return 5;
  case gpu::Processor::ThreadX:
    return 6;
  case gpu::Processor::ThreadY:
    return 7;
  case gpu::Processor::ThreadZ:
    return 8;
  default:;
  }
  llvm_unreachable(
      "invalid processor type while retrieving launch op argument number");
}

bool checkIfCopyLoop(ParallelOp parallelOp) {
  BoolAttr copyLoop = parallelOp->getAttrOfType<BoolAttr>("isCopyLoopNest");
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

  // Create ops for dimensions of thread block.
  for (int64_t param : tbDims) {
    tbDimValues.push_back(rewriter.create<ConstantIndexOp>(topLoc, param));
  }

  /////////////////////////////////////////////////////////////////////////////
  // Create ops for dimensions of grid. The grid dimensions are dependent on the
  // dimensions of the thread block and the number of iteratoins of the loop
  // being mapped. Currently making a simple assumption that one thread block
  // will calculate the	loopStep amount of iterations. Hence the number of
  // thread blocks to be launched will be (loopUB - loopLB) / loopStep for a
  // particular loop.
  /////////////////////////////////////////////////////////////////////////////

  // Create Ops for dimensions of grid. The number of thread blocks to be
  // launched will be (loopUB + loopStep - 1) / loopStep.
  Value constantOne = rewriter.create<ConstantIndexOp>(topLoc, 1);
  for (auto loop : llvm::zip(parallelOp.upperBound(), parallelOp.step())) {
    Value upperBound, step;
    std::tie(upperBound, step) = loop;
    Value resultA = rewriter.create<AddIOp>(topLoc, upperBound, step);
    Value resultB = rewriter.create<SubIOp>(topLoc, resultA, constantOne);
    gridDimValues.insert(gridDimValues.begin(), rewriter.create<UnsignedDivIOp>(
                                                    topLoc, resultB, step));
    /////////////////////////////////////////////////////////////////////////////
    // Check if the defining ops for upper/lower bound and step are constant.
    // If not the return false.
    // if (isa<ConstantIndexOp>(lowerBound.getDefiningOp()) &&
    //    isa<ConstantIndexOp>(upperBound.getDefiningOp()) &&
    //    isa<ConstantIndexOp>(step.getDefiningOp())) {
    //  Value diff = rewriter.create<SubIOp>(topLoc, upperBound, lowerBound);
    //  gridDimValues.push_back(
    //      rewriter.create<SignedDivIOp>(topLoc, diff, step));
    //} else
    //  return false;
    /////////////////////////////////////////////////////////////////////////////
  }
  return true;
}

LogicalResult convertIfOp(gpu::LaunchOp launchOp, IfOp ifOp,
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
                             BlockAndValueMapping &cloningMap,
                             SmallVectorImpl<Operation *> &worklist,
                             PatternRewriter &rewriter) {
  Location loc = forOp.getLoc();
  scf::ForOp loopOp;
  if (forOp.getNumResults() > 0) {
    SmallVector<Value, 4> loopOpIterArgs;
    for (auto args : forOp.getIterOperands())
      loopOpIterArgs.push_back(cloningMap.lookupOrDefault(args));
    loopOp = rewriter.create<scf::ForOp>(
        loc, cloningMap.lookupOrDefault(forOp.lowerBound()),
        cloningMap.lookupOrDefault(forOp.upperBound()),
        cloningMap.lookupOrDefault(forOp.step()), loopOpIterArgs,
        [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
          builder.create<scf::YieldOp>(loc, iterArgs);
        });
    auto &forYieldOp = forOp.getLoopBody().front().back();
    auto yieldOperands = forYieldOp.getOperands();
    SmallVector<Value, 4> loopOpYieldOper;
    for (auto oper : yieldOperands)
      loopOpYieldOper.push_back(cloningMap.lookupOrDefault(oper));
    loopOp.getLoopBody().front().back().setOperands(loopOpYieldOper);
    cloningMap.map(forOp.getLoopBody().getArguments(),
                   loopOp.getLoopBody().getArguments());
    cloningMap.map(forOp.getResults(), loopOp.getResults());
  } else {
    loopOp = rewriter.create<scf::ForOp>(
        loc, cloningMap.lookupOrDefault(forOp.lowerBound()),
        cloningMap.lookupOrDefault(forOp.upperBound()),
        cloningMap.lookupOrDefault(forOp.step()));
  }

  Value newIndex = loopOp.getInductionVar();
  rewriter.setInsertionPointToStart(loopOp.getBody());
  // Put a sentinel into the worklist so we know when to pop out of the loop
  // body again. We use the launchOp here, as that cannot be part of the bodies
  // instruction.
  worklist.push_back(launchOp.getOperation());
  cloningMap.map(forOp.getInductionVar(), newIndex);

  Block *body = forOp.getBody();
  worklist.reserve(worklist.size() + body->getOperations().size());

  if (forOp.getNumResults() > 0)
    worklist.push_back(&loopOp.getLoopBody().front().back());
  for (Operation &op : llvm::reverse(body->without_terminator()))
    worklist.push_back(&op);
  return success();
}

LogicalResult convertParallelLoop(gpu::LaunchOp launchOp, ParallelOp parallelOp,
                                  BlockAndValueMapping &cloningMap,
                                  SmallVectorImpl<Operation *> &worklist,
                                  PatternRewriter &rewriter) {
  Location loc = parallelOp.getLoc();
  if (checkIfCopyLoop(parallelOp)) {
    assert(parallelOp.getNumLoops() == 1 && "Expected a 1-d copy loop.");
    // Copy loops are handeled specially. A copy loop is assumed to be 1-d and
    // is distributed among the threads in a linear fashion so as to enable
    // global memory coallescing.
    // TODO: Enable further optimizations such as prevention of shared memory
    // bank conflicts while loading the operands.
    for (auto loop : llvm::zip(parallelOp.getInductionVars(),
                               parallelOp.upperBound(), parallelOp.step())) {
      Value iv, upperBound, step;
      std::tie(iv, upperBound, step) = loop;

      auto loopOp = rewriter.create<scf::ForOp>(
          loc, linearTidXYZ, cloningMap.lookupOrDefault(upperBound),
          numThreadsXYZ);

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
        parallelOp->getAttrOfType<ArrayAttr>(gpu::getMappingAttrName());
    // Check if mapping attribute is present or not.
    if (!mapping)
      return parallelOp.emitOpError("expected mapping attribute");

    Value numWarpsInN, numWarpsInM, warpIdX, warpIdY;

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
      // Checks if the loop is mapped to some processor or it is sequental.
      if (isMappedToProcessor(processor)) {
        // Checks if the loop is mapped to a grid.
        if (processor < gpu::Processor::WarpX) {
          // Use the corresponding grid index as replacement for the loop
          // iv.
          Value operand =
              launchOp.body().getArgument(getLaunchOpArgumentNum(processor));
          Value mulWithStep = rewriter.create<MulIOp>(loc, operand, step);
          Value newIV = rewriter.create<AddIOp>(loc, lowerBound, mulWithStep);
          cloningMap.map(iv, newIV);
        } else if (processor < gpu::Processor::ThreadX) {
          Value loopOpLB, loopOpUB, loopOpStep;
          if (processor == gpu::Processor::WarpY) {
            // SmallVector<Value, 4> loopSteps(parallelOp.step());
            // warpMtile = loopSteps[0];
            // warpNtile = loopSteps[1];

            Value divNtileByWarpNtile =
                rewriter.create<UnsignedDivIOp>(loc, nTile, warpNtile);
            Value cmpResult = rewriter.create<CmpIOp>(
                loc, CmpIPredicate::ule, numWarps, divNtileByWarpNtile);

            numWarpsInN = rewriter.create<SelectOp>(loc, cmpResult, numWarps,
                                                    divNtileByWarpNtile);
            numWarpsInM =
                rewriter.create<UnsignedDivIOp>(loc, numWarps, numWarpsInN);

            warpIdX =
                rewriter.create<UnsignedRemIOp>(loc, linearWarpId, numWarpsInN);
            warpIdY =
                rewriter.create<UnsignedDivIOp>(loc, linearWarpId, numWarpsInN);

            loopOpLB = rewriter.create<MulIOp>(loc, warpIdY, warpMtile);
            loopOpUB = mTile;
            loopOpStep = rewriter.create<MulIOp>(loc, warpMtile, numWarpsInM);

          } else if (processor == gpu::Processor::WarpX) {

            loopOpLB = rewriter.create<MulIOp>(loc, warpIdX, warpNtile);
            loopOpUB = nTile;
            loopOpStep = rewriter.create<MulIOp>(loc, warpNtile, numWarpsInN);
          }

          ForOp loopOp =
              rewriter.create<ForOp>(loc, loopOpLB, loopOpUB, loopOpStep);
          Value newIndex = loopOp.getInductionVar();
          rewriter.setInsertionPointToStart(loopOp.getBody());
          // Put a sentinel into the worklist so we know when to pop out of the
          // loop body again. We use the launchOp here, as that cannot be part
          // of the bodies instruction.
          worklist.push_back(launchOp.getOperation());
          cloningMap.map(iv, newIndex);
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
              launchOp.body().getArgument(getLaunchOpArgumentNum(processor) -
                                          3),
              cloningMap.lookupOrDefault(upperBound),
              cloningMap.lookupOrDefault(
                  launchOp.getOperand(getLaunchOpArgumentNum(processor) - 3)));
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

// Doing pre computation stuff i.e. computing linear thread id, linear warp id,
// number of threads.
static void doPreComputationStuff(gpu::LaunchOp gpuLaunchOp,
                                  ParallelOp parallelOp,
                                  PatternRewriter &rewriter) {
  assert(parallelOp.getNumLoops() >= 2 && "expected atleast a 2-d loop nest");
  Location loc = parallelOp.getLoc();

  // Find linear thread id.
  // Insert ops for calculating the linear thread Id.
  Value xdimYdim = rewriter.create<MulIOp>(loc, gpuLaunchOp.blockSizeX(),
                                           gpuLaunchOp.blockSizeY());
  Value zIdXdimYdim =
      rewriter.create<MulIOp>(loc, gpuLaunchOp.getThreadIds().z, xdimYdim);
  Value yIdXdim = rewriter.create<MulIOp>(loc, gpuLaunchOp.getThreadIds().y,
                                          gpuLaunchOp.blockSizeX());
  Value linearTidYZ = rewriter.create<AddIOp>(loc, zIdXdimYdim, yIdXdim);
  linearTidXYZ =
      rewriter.create<AddIOp>(loc, linearTidYZ, gpuLaunchOp.getThreadIds().x);

  // numThreadsXYZ = rewriter.create<MulIOp>(loc, xdimYdim,
  // gpuLaunchOp.blockSizeZ());

  Value constantWarpSize = rewriter.create<ConstantIndexOp>(loc, warpSize);
  linearWarpId =
      rewriter.create<UnsignedDivIOp>(loc, linearTidXYZ, constantWarpSize);

  numWarps =
      rewriter.create<UnsignedDivIOp>(loc, numThreadsXYZ, constantWarpSize);
}

static void computeNumThreads(Location loc, PatternRewriter &rewriter) {
  Value mbywarpM = rewriter.create<UnsignedDivIOp>(loc, mTile, warpMtile);
  Value nbywarpN = rewriter.create<UnsignedDivIOp>(loc, nTile, warpNtile);
  Value mbyWMintonbyWN = rewriter.create<MulIOp>(loc, mbywarpM, nbywarpN);
  Value constantWarpSize = rewriter.create<ConstantIndexOp>(loc, warpSize);
  numThreadsXYZ =
      rewriter.create<MulIOp>(loc, mbyWMintonbyWN, constantWarpSize);
}

static void findTileSizes(ParallelOp parallelOp) {
  ParallelOp warpLoop;
  parallelOp.walk([&](ParallelOp op) {
    if ((!op->getAttrOfType<BoolAttr>("isCopyLoopNest")) &&
        (op.getNumLoops() == 2) && op != parallelOp) {
      ArrayAttr mapping =
          op->getAttrOfType<ArrayAttr>(gpu::getMappingAttrName());
      for (auto attr : mapping) {
        auto annotation = attr.dyn_cast<gpu::ParallelLoopDimMapping>();
        if ((gpu::getProcessor(annotation) > gpu::Processor::BlockZ) &&
            (gpu::getProcessor(annotation) < gpu::Processor::ThreadX)) {
          warpLoop = op;
          return;
        }
      }
    }
  });

  assert(warpLoop.getNumLoops() == 2 && "Not a 2-d warp loop");

  SmallVector<Value, 2> threadBlockLoopSteps(parallelOp.step());
  SmallVector<Value, 2> warpLoopSteps(warpLoop.step());

  mTile = threadBlockLoopSteps[0];
  nTile = threadBlockLoopSteps[1];
  warpMtile = warpLoopSteps[0];
  warpNtile = warpLoopSteps[1];
}

LogicalResult
LoopsToGpuLowering::matchAndRewrite(ParallelOp parallelOp,
                                    PatternRewriter &rewriter) const {
  Location topLoc = parallelOp.getLoc();
  Value constantOne = rewriter.create<ConstantIndexOp>(topLoc, 1);
  SmallVector<Value, 3> tbDimValues, gridDimValues;
  // Computing GPU launch block grid and thread block dimensions.
  if (!insertLaunchParams(parallelOp, tbDims, rewriter, topLoc, tbDimValues,
                          gridDimValues))
    return failure();
  gridDimValues.insert(gridDimValues.end(), 3 - gridDimValues.size(),
                       constantOne);

  findTileSizes(parallelOp);
  computeNumThreads(topLoc, rewriter);
  gpu::LaunchOp launchOp = rewriter.create<gpu::LaunchOp>(
      parallelOp.getLoc(), gridDimValues[0], gridDimValues[1], gridDimValues[2],
      numThreadsXYZ, tbDimValues[1], tbDimValues[2]);

  rewriter.setInsertionPointToEnd(&launchOp.body().front());
  rewriter.create<gpu::TerminatorOp>(topLoc);
  rewriter.setInsertionPointToStart(&launchOp.body().front());

  // Doing Pre Computation Stuff.
  doPreComputationStuff(launchOp, parallelOp, rewriter);

  BlockAndValueMapping cloningMap;
  SmallVector<Operation *, 16> worklist;
  if (failed(convertParallelLoop(launchOp, parallelOp, cloningMap, worklist,
                                 rewriter)))
    return failure();

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (auto nestedParallel = dyn_cast<ParallelOp>(op)) {
      if (failed(convertParallelLoop(launchOp, nestedParallel, cloningMap,
                                     worklist, rewriter)))
        return failure();
    } else if (op == launchOp.getOperation()) {
      auto parent = rewriter.getInsertionPoint()->getParentOp();
      rewriter.setInsertionPointAfter(parent);
    } else if (auto nestedFor = dyn_cast<ForOp>(op)) {
      if (failed(convertForLoop(launchOp, nestedFor, cloningMap, worklist,
                                rewriter)))
        return failure();
    } else if (auto nestedIf = dyn_cast<IfOp>(op)) {
      if (nestedIf->getParentOfType<gpu::LaunchOp>() == launchOp) {
        // This is a sentinal op. Set the rewriter to the then part of the if
        // op.
        if (IfOp parent =
                dyn_cast<IfOp>(rewriter.getInsertionPoint()->getParentOp()))
          rewriter.setInsertionPointToStart(&parent.elseRegion().front());
      } else {
        if (failed(convertIfOp(launchOp, nestedIf, cloningMap, worklist,
                               rewriter)))
          return failure();
      }
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      int count = 0;
      for (auto yieldOper : yieldOp.getOperands()) {
        yieldOp.setOperand(count, cloningMap.lookupOrDefault(yieldOper));
        count++;
      }
    } else {
      Operation *clone = rewriter.clone(*op, cloningMap);
      cloningMap.map(op->getResults(), clone->getResults());
    }
  }
  rewriter.eraseOp(parallelOp);
  return success();
}

void populateConvertSCFToGPUPatterns(OwningRewritePatternList &patterns,
                                     MLIRContext *ctx,
                                     ArrayRef<int64_t> tbDims) {
  patterns.insert<LoopsToGpuLowering>(ctx, tbDims);
}

void TestConvertSCFToGPUPass::filltbDims() {
  for (int i = 0, e = tbDimsRef.size(); i < 3; ++i) {
    if (i < e)
      tbDims.push_back(tbDimsRef[i]);
    else
      tbDims.push_back(1);
  }
}

void TestConvertSCFToGPUPass::runOnOperation() {
  warpSize = warpSizeRef;
  filltbDims();
  OwningRewritePatternList patterns;
  populateConvertSCFToGPUPatterns(patterns, &getContext(), tbDims);
  ConversionTarget target(getContext());
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<AffineDialect>();
  target.addLegalDialect<gpu::GPUDialect>();
  target.addLegalDialect<scf::SCFDialect>();
  target.addIllegalOp<scf::ParallelOp>();
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
void registerTestConvertSCFToGPUPass() {
  PassRegistration<TestConvertSCFToGPUPass>("test-convert-scf-to-gpu",
                                            "Convert SCF to GPU");
}
} // namespace mlir
