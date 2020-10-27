//===- SCFToGPU.cpp - Convert an affine loop nest to a GPU kernel -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements a straightforward conversion of an loop nest into a GPU
// kernel.  The caller is expected to guarantee that the conversion is correct
// or to further transform the kernel to ensure correctness.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/ParallelLoopMapper.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "loops-to-gpu"

using namespace mlir;
using namespace mlir::scf;

// Extract an indexed value from KernelDim3.
static Value getDim3Value(const gpu::KernelDim3 &dim3, unsigned pos) {
  switch (pos) {
  case 0:
    return dim3.x;
  case 1:
    return dim3.y;
  case 2:
    return dim3.z;
  default:
    llvm_unreachable("dim3 position out of bounds");
  }
  return nullptr;
}

// Get the lower bound-related operands of a loop operation.
static Operation::operand_range getLowerBoundOperands(AffineForOp forOp) {
  return forOp.getLowerBoundOperands();
}

// Get the upper bound-related operands of a loop operation.
static Operation::operand_range getUpperBoundOperands(AffineForOp forOp) {
  return forOp.getUpperBoundOperands();
}

// Get a Value that corresponds to the loop step.  If the step is an attribute,
// materialize a corresponding constant using builder.
static Value getOrCreateStep(AffineForOp forOp, OpBuilder &builder) {
  return builder.create<ConstantIndexOp>(forOp.getLoc(), forOp.getStep());
}

// Get a Value for the loop lower bound.  If the value requires computation,
// materialize the instructions using builder.
static Value getOrEmitLowerBound(AffineForOp forOp, OpBuilder &builder) {
  return lowerAffineLowerBound(forOp, builder);
}

// Get a Value for the loop upper bound.  If the value requires computation,
// materialize the instructions using builder.
static Value getOrEmitUpperBound(AffineForOp forOp, OpBuilder &builder) {
  return lowerAffineUpperBound(forOp, builder);
}

// Check the structure of the loop nest:
//   - there are enough loops to map to numDims;
//   - the loops are perfectly nested;
//   - the loop bounds can be computed above the outermost loop.
// This roughly corresponds to the "matcher" part of the pattern-based
// rewriting infrastructure.
static LogicalResult checkAffineLoopNestMappableImpl(AffineForOp forOp,
                                                     unsigned numDims) {
  Region &limit = forOp.region();
  for (unsigned i = 0, e = numDims; i < e; ++i) {
    Operation *nested = &forOp.getBody()->front();
    if (!areValuesDefinedAbove(getLowerBoundOperands(forOp), limit) ||
        !areValuesDefinedAbove(getUpperBoundOperands(forOp), limit))
      return forOp.emitError(
          "loops with bounds depending on other mapped loops "
          "are not supported");

    // The innermost loop can have an arbitrary body, skip the perfect nesting
    // check for it.
    if (i == e - 1)
      break;

    auto begin = forOp.getBody()->begin(), end = forOp.getBody()->end();
    if (forOp.getBody()->empty() || std::next(begin, 2) != end)
      return forOp.emitError("expected perfectly nested loops in the body");

    if (!(forOp = dyn_cast<AffineForOp>(nested)))
      return nested->emitError("expected a nested loop");
  }
  return success();
}

static LogicalResult checkAffineLoopNestMappable(AffineForOp forOp,
                                                 unsigned numBlockDims,
                                                 unsigned numThreadDims) {
  if (numBlockDims < 1 || numThreadDims < 1) {
    LLVM_DEBUG(llvm::dbgs() << "nothing to map");
    return success();
  }

  if (numBlockDims > 3) {
    return forOp.emitError("cannot map to more than 3 block dimensions");
  }
  if (numThreadDims > 3) {
    return forOp.emitError("cannot map to more than 3 thread dimensions");
  }
  return checkAffineLoopNestMappableImpl(forOp, numBlockDims + numThreadDims);
}

namespace {
// Helper structure that holds common state of the loop to GPU kernel
// conversion.
struct AffineLoopToGpuConverter {
  Optional<AffineForOp> collectBounds(AffineForOp forOp, unsigned numLoops);

  void createLaunch(AffineForOp rootForOp, AffineForOp innermostForOp,
                    unsigned numBlockDims, unsigned numThreadDims);

  // Ranges of the loops mapped to blocks or threads.
  SmallVector<Value, 6> dims;
  // Lower bounds of the loops mapped to blocks or threads.
  SmallVector<Value, 6> lbs;
  // Induction variables of the loops mapped to blocks or threads.
  SmallVector<Value, 6> ivs;
  // Steps of the loops mapped to blocks or threads.
  SmallVector<Value, 6> steps;
};
} // namespace

// Return true if the value is obviously a constant "one".
static bool isConstantOne(Value value) {
  if (auto def = value.getDefiningOp<ConstantIndexOp>())
    return def.getValue() == 1;
  return false;
}

// Collect ranges, bounds, steps and induction variables in preparation for
// mapping a loop nest of depth "numLoops" rooted at "forOp" to a GPU kernel.
// This may fail if the IR for computing loop bounds cannot be constructed, for
// example if an affine loop uses semi-affine maps. Return the last loop to be
// mapped on success, llvm::None on failure.
Optional<AffineForOp>
AffineLoopToGpuConverter::collectBounds(AffineForOp forOp, unsigned numLoops) {
  OpBuilder builder(forOp.getOperation());
  dims.reserve(numLoops);
  lbs.reserve(numLoops);
  ivs.reserve(numLoops);
  steps.reserve(numLoops);
  AffineForOp currentLoop = forOp;
  for (unsigned i = 0; i < numLoops; ++i) {
    Value lowerBound = getOrEmitLowerBound(currentLoop, builder);
    Value upperBound = getOrEmitUpperBound(currentLoop, builder);
    if (!lowerBound || !upperBound) {
      return llvm::None;
    }

    Value range =
        builder.create<SubIOp>(currentLoop.getLoc(), upperBound, lowerBound);
    Value step = getOrCreateStep(currentLoop, builder);
    if (!isConstantOne(step))
      range = builder.create<SignedDivIOp>(currentLoop.getLoc(), range, step);
    dims.push_back(range);

    lbs.push_back(lowerBound);
    ivs.push_back(currentLoop.getInductionVar());
    steps.push_back(step);

    if (i != numLoops - 1)
      currentLoop = cast<AffineForOp>(&currentLoop.getBody()->front());
  }
  return currentLoop;
}

// Replace the rooted at "rootForOp" with a GPU launch operation.  This expects
// "innermostForOp" to point to the last loop to be transformed to the kernel,
// and to have (numBlockDims + numThreadDims) perfectly nested loops between
// "rootForOp" and "innermostForOp".
void AffineLoopToGpuConverter::createLaunch(AffineForOp rootForOp,
                                            AffineForOp innermostForOp,
                                            unsigned numBlockDims,
                                            unsigned numThreadDims) {
  OpBuilder builder(rootForOp.getOperation());
  // Prepare the grid and block sizes for the launch operation.  If there is
  // no loop mapped to a specific dimension, use constant "1" as its size.
  Value constOne = (numBlockDims < 3 || numThreadDims < 3)
                       ? builder.create<ConstantIndexOp>(rootForOp.getLoc(), 1)
                       : nullptr;
  Value gridSizeX = numBlockDims > 0 ? dims[0] : constOne;
  Value gridSizeY = numBlockDims > 1 ? dims[1] : constOne;
  Value gridSizeZ = numBlockDims > 2 ? dims[2] : constOne;
  Value blockSizeX = numThreadDims > 0 ? dims[numBlockDims] : constOne;
  Value blockSizeY = numThreadDims > 1 ? dims[numBlockDims + 1] : constOne;
  Value blockSizeZ = numThreadDims > 2 ? dims[numBlockDims + 2] : constOne;

  // Create a launch op and move the body region of the innermost loop to the
  // launch op.
  auto launchOp = builder.create<gpu::LaunchOp>(
      rootForOp.getLoc(), gridSizeX, gridSizeY, gridSizeZ, blockSizeX,
      blockSizeY, blockSizeZ);

  // Replace the loop terminator (loops contain only a single block) with the
  // gpu terminator and move the operations from the loop body block to the gpu
  // launch body block.  Do not move the entire block because of the difference
  // in block arguments.
  Operation &terminator = innermostForOp.getBody()->back();
  Location terminatorLoc = terminator.getLoc();
  terminator.erase();
  builder.setInsertionPointToEnd(innermostForOp.getBody());
  builder.create<gpu::TerminatorOp>(terminatorLoc, llvm::None);
  launchOp.body().front().getOperations().splice(
      launchOp.body().front().begin(),
      innermostForOp.getBody()->getOperations());

  // Remap the loop iterators to use block/thread identifiers instead.  Loops
  // may iterate from LB with step S whereas GPU thread/block ids always iterate
  // from 0 to N with step 1.  Therefore, loop induction variables are replaced
  // with (gpu-thread/block-id * S) + LB.
  builder.setInsertionPointToStart(&launchOp.body().front());
  auto lbArgumentIt = lbs.begin();
  auto stepArgumentIt = steps.begin();
  for (auto en : llvm::enumerate(ivs)) {
    Value id =
        en.index() < numBlockDims
            ? getDim3Value(launchOp.getBlockIds(), en.index())
            : getDim3Value(launchOp.getThreadIds(), en.index() - numBlockDims);
    Value step = steps[en.index()];
    if (!isConstantOne(step))
      id = builder.create<MulIOp>(rootForOp.getLoc(), step, id);

    Value ivReplacement =
        builder.create<AddIOp>(rootForOp.getLoc(), *lbArgumentIt, id);
    en.value().replaceAllUsesWith(ivReplacement);
    std::advance(lbArgumentIt, 1);
    std::advance(stepArgumentIt, 1);
  }

  // We are done and can erase the original outermost loop.
  rootForOp.erase();
}

// Generic loop to GPU kernel conversion function.
static LogicalResult convertAffineLoopNestToGPULaunch(AffineForOp forOp,
                                                      unsigned numBlockDims,
                                                      unsigned numThreadDims) {
  if (failed(checkAffineLoopNestMappable(forOp, numBlockDims, numThreadDims)))
    return failure();

  AffineLoopToGpuConverter converter;
  auto maybeInnerLoop =
      converter.collectBounds(forOp, numBlockDims + numThreadDims);
  if (!maybeInnerLoop)
    return failure();
  converter.createLaunch(forOp, *maybeInnerLoop, numBlockDims, numThreadDims);

  return success();
}

LogicalResult mlir::convertAffineLoopNestToGPULaunch(AffineForOp forOp,
                                                     unsigned numBlockDims,
                                                     unsigned numThreadDims) {
  return ::convertAffineLoopNestToGPULaunch(forOp, numBlockDims, numThreadDims);
}

namespace {
struct ParallelToGpuLaunchLowering : public OpRewritePattern<ParallelOp> {
  using OpRewritePattern<ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ParallelOp parallelOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace

/// Checks if the loop bounds are dependent on the loop IV of an outer loop
/// which has laready been mapped to a H/W ID.
static bool getGpuMappedDefiningOp(Value &toFind, gpu::LaunchOp &launchOp,
                                   llvm::DenseMap<Value, bool> &mappedloopIVs,
                                   Value *mappedLoopIv = nullptr) {
  // If the value is defined outside the gpu.launch then return false.
  if (toFind.getParentRegion()->isAncestor(launchOp.getParentRegion()))
    return false;

  //// Check If the definingOp is a constant, if yes then just return.
  // if (isa_and_nonnull<ConstantOp>(toFind.getDefiningOp()))
  //  return false;

  // Check if `toFind` is mapped to a H/W ID.
  if (mappedloopIVs.find(toFind) != mappedloopIVs.end()) {
    if (mappedLoopIv) 
      *mappedLoopIv = toFind;
    
    return true;
  }

  bool isMapped = false;
  if (!toFind.getParentRegion()->isAncestor(launchOp.getParentRegion())) {
    // Try to find if the op is mapped to the GPU H/W ID recursively.
    for (Value operand : toFind.getDefiningOp()->getOperands()) {
      isMapped |= getGpuMappedDefiningOp(operand, launchOp, mappedloopIVs,
                                         mappedLoopIv);
    }
  }

  return isMapped;
}

/// Tries to create a series of Operations which approximate the upper bound of
/// of a loop that has original upper bound dependent on a loop IV which has
/// been mapped to a GPU H/W ID. Currently, just use the step of the mapped loop
/// and create a constantOp which can be used as the number of iterations.
static void
approximateIterations(Value &loopBoundIV, Value &step, gpu::LaunchOp &launchOp,
                      llvm::DenseMap<Value, bool> &mappedloopIVs,
                      SmallVectorImpl<ParallelOp> &mappedParallelLoops,
                      PatternRewriter &rewriter, Value &launchBound, Value & outerloopStep) {
  // Check if the defining op of the upper bound of this loop is a
  // constantIndexop. Find the loop which has already been mapped
  // to a GPU H/W ID and has the same loop IV as the one used in
  // the upper bound.
  for (ParallelOp parallelOp : mappedParallelLoops) {
    for (auto loop :
         llvm::zip(parallelOp.getInductionVars(), parallelOp.step())) {
      if (isa<ConstantIndexOp>(std::get<1>(loop).getDefiningOp()) &&
          std::get<0>(loop) == loopBoundIV) {
	outerloopStep = std::get<1>(loop).getDefiningOp()->getResult(0);
        Operation *curLoopStep = rewriter.clone(*(step.getDefiningOp()));
	launchBound = rewriter.create<SignedDivIOp>(
            parallelOp.getLoc(), std::get<1>(loop), curLoopStep->getResult(0));
	return;
      }
    }
  }
}

/// Tries to derive a static upper bound from the defining operation of
/// `upperBound`.
static Value deriveStaticUpperBound(Value upperBound,
                                    PatternRewriter &rewriter) {
  if (auto op = upperBound.getDefiningOp<ConstantIndexOp>()) {
    return op;
  }

  if (auto minOp = upperBound.getDefiningOp<AffineMinOp>()) {
    for (const AffineExpr &result : minOp.map().getResults()) {
      if (auto constExpr = result.dyn_cast<AffineConstantExpr>()) {
        return rewriter.create<ConstantIndexOp>(minOp.getLoc(),
                                                constExpr.getValue());
      }
    }
  }

  if (auto multiplyOp = upperBound.getDefiningOp<MulIOp>()) {
    if (auto lhs = dyn_cast_or_null<ConstantIndexOp>(
            deriveStaticUpperBound(multiplyOp.getOperand(0), rewriter)
                .getDefiningOp()))
      if (auto rhs = dyn_cast_or_null<ConstantIndexOp>(
              deriveStaticUpperBound(multiplyOp.getOperand(1), rewriter)
                  .getDefiningOp())) {
        // Assumptions about the upper bound of minimum computations no longer
        // work if multiplied by a negative value, so abort in this case.
        if (lhs.getValue() < 0 || rhs.getValue() < 0)
          return {};

        return rewriter.create<ConstantIndexOp>(
            multiplyOp.getLoc(), lhs.getValue() * rhs.getValue());
      }
  }

  return {};
}

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

/// Modifies the current transformation state to capture the effect of the given
/// `scf.parallel` operation on index substitutions and the operations to be
/// inserted.
/// Specifically, if a dimension of a parallel loop is mapped to a hardware id,
/// this function will
/// - compute the loop index based on the hardware id and affine map from the
///   mapping and update `cloningMap` to substitute all uses.
/// - derive a new upper bound for the hardware id and augment the provided
///   `gpu.launch operation` accordingly.
/// - if the upper bound is imprecise, insert a conditional in the `gpu.launch`
///   and update the rewriter to insert into the conditional's body.
/// If the dimension is mapped to sequential,
/// - insert a for loop into the body and update the rewriter to insert into
///   the for loop's body.
/// - update the `cloningMap` to replace uses of the index with the index of
///   the new for loop.
/// In either case,
/// - append the instructions from the loops body to worklist, in reverse order.
/// To note the end of the current scope in case a loop or conditional was
/// inserted, a sentinel (the `gpu.launch` operation) is inserted into the
/// worklist. This signals the processor of the worklist to pop the rewriter
/// one scope-level up.
static LogicalResult processParallelLoop(
    ParallelOp parallelOp, gpu::LaunchOp launchOp,
    BlockAndValueMapping &cloningMap, SmallVectorImpl<Operation *> &worklist,
    DenseMap<gpu::Processor, Value> &bounds,
    SmallVectorImpl<ParallelOp> &mappedParallelLoops,
    llvm::DenseMap<Value, bool> &mappedloopIVs, llvm::DenseMap<Value, Value> &loopIVtoIter, PatternRewriter &rewriter) {
  // TODO: Verify that this is a valid GPU mapping.
  // processor ids: 0-2 block [x/y/z], 3-5 -> thread [x/y/z], 6-> sequential
  ArrayAttr mapping =
      parallelOp.getAttrOfType<ArrayAttr>(gpu::getMappingAttrName());

  // TODO: Support reductions.
  if (!mapping || parallelOp.getNumResults() != 0)
    return failure();

  Location loc = parallelOp.getLoc();

  auto launchIndependent = [&launchOp](Value val) {
    return val.getParentRegion()->isAncestor(launchOp.getParentRegion());
  };

  auto ensureLaunchIndependent = [&rewriter,
                                  launchIndependent](Value val) -> Value {
    if (launchIndependent(val))
      return val;
    if (ConstantOp constOp = val.getDefiningOp<ConstantOp>())
      return rewriter.create<ConstantOp>(constOp.getLoc(), constOp.getValue());
    return {};
  };

  for (auto config : llvm::zip(mapping, parallelOp.getInductionVars(),
                               parallelOp.lowerBound(), parallelOp.upperBound(),
                               parallelOp.step())) {
    Attribute mappingAttribute;
    Value iv, lowerBound, upperBound, step;
    std::tie(mappingAttribute, iv, lowerBound, upperBound, step) = config;
    auto annotation = mappingAttribute.dyn_cast<gpu::ParallelLoopDimMapping>();
    if (!annotation)
      return parallelOp.emitOpError()
             << "expected mapping attribute for lowering to GPU";
    Value newIndex;
    gpu::Processor processor = gpu::getProcessor(annotation);

    if (isMappedToProcessor(processor)) {
      // If there was also a bound, insert that, too.
      // TODO: Check that we do not assign bounds twice.
      if (annotation.bound().getValue()) {
        // We pass as the single operand to the bound-map the number of
        // iterations, which is (upperBound - lowerBound) ceilDiv step. To
        // support inner loops with dynamic upper bounds (as generated by e.g.
        // tiling), try to derive a max for the bounds. If the used bound for
        // the hardware id is imprecise, wrap the contained code into a
        // conditional. If the lower-bound is constant or defined before the
        // launch, we can use it in the launch bounds. Otherwise fail.
        // TODO: We dont have to fail in each and every case. The lower bound
        // can very well be dependent on the outer loop IV. If the outer loop IV
        // has been already mapped to a GPU H/W ID then it stays constant
        // in a particular instance of the gpu.launch op, We can use that H/W
        // ID isntead of jsut failing.
        if (!launchIndependent(lowerBound) &&
            !isa_and_nonnull<ConstantOp>(lowerBound.getDefiningOp())) {
          if (!getGpuMappedDefiningOp(lowerBound, launchOp, mappedloopIVs)) {
            return failure();
          }
        }
        // The step must also be constant or defined outside of the loop nest.
        if (!launchIndependent(step) &&
            !isa_and_nonnull<ConstantOp>(step.getDefiningOp())) {
          return failure();
        }
        // If the upper-bound is constant or defined before the launch, we can
        // use it in the launch bounds directly. Otherwise try derive a bound.
        // The upper bound can also be dependent on an outer loop IV which has
        // already been mapped to a H/W ID in which case we can also try to
        // approximate the bound.
        bool boundIsPrecise =
            launchIndependent(upperBound) ||
            isa_and_nonnull<ConstantOp>(upperBound.getDefiningOp());
        bool toApproximateIterations = false;
        {
          PatternRewriter::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(launchOp);
          Value lowerBoundIV, upperBoundIV;
          if (!boundIsPrecise) {
            // Check if upper bound can be obtianed.
	    Value tempUpperBound = deriveStaticUpperBound(upperBound, rewriter);
            if (!tempUpperBound) {
              // If upper bound is not statically obtainable check if both the
              // upper and lower bound are outer loopIV dependent.
              if (getGpuMappedDefiningOp(lowerBound, launchOp, mappedloopIVs,
                                         &lowerBoundIV) &&
                  getGpuMappedDefiningOp(upperBound, launchOp, mappedloopIVs,
                                         &upperBoundIV)) {
                if (lowerBoundIV != upperBoundIV)
                  // If both are outer loopIV dependent but are not dependent
                  // on the saame loopIV then abort.
                  return failure();
                else
                  toApproximateIterations = true;
              } else
                // If both are not dependent on the outer loopIV then abort.
                return failure();
            }
          } else {
            // Check for the case when uppper bound is aproximated but the lower
            // bound is dependent on the outer loopIV. abort in this case.
            if (getGpuMappedDefiningOp(lowerBound, launchOp, mappedloopIVs))
              return failure();
          }
          Value launchBound;
          if (toApproximateIterations) {
	    Value outerloopStep;
            // Try to approximate the number of iterations if the upper and
            // lower bound is dependent on the same loop IV.
            approximateIterations(upperBoundIV, step, launchOp, mappedloopIVs,
                                  mappedParallelLoops, rewriter, launchBound, outerloopStep);
        
      // Use the corresponding thread/grid index as replacement for the loop iv.
      //if(loopIVtoIter.find(lowerBoundIV) == loopIVtoIter.end())
      //  return failure();
      //Value operand =
      //    launchOp.body().getArgument(getLaunchOpArgumentNum(processor));
      //// Take the indexmap and add the lower bound and step computations in.
      //// This computes operand * step + OuterLoopIv * outerLoopStep.
      //// Use an affine map here so that it composes nicely with the provided
      //// annotation.
      //AffineMap lowerAndStep = AffineMap::get(
      //    1, 3,
      //    rewriter.getAffineDimExpr(0) * rewriter.getAffineSymbolExpr(0) +
      //        (rewriter.getAffineSymbolExpr(1)) * rewriter.getAffineSymbolExpr(2));
      //newIndex = rewriter.create<AffineApplyOp>(
      //    loc, annotation.map().getValue().compose(lowerAndStep),
      //    ValueRange{operand, step, loopIVtoIter[lowerBoundIV], outerloopStep});
          } else {
            // Compute the number of iterations needed. We compute this as an
            // affine expression ceilDiv (upperBound - lowerBound) step. We use
            // affine.apply here so that it composes nicely with the provided
            // map.
            AffineMap stepMap =
                AffineMap::get(0, 3,
                               ((rewriter.getAffineSymbolExpr(0) -
                                 rewriter.getAffineSymbolExpr(1))
                                    .ceilDiv(rewriter.getAffineSymbolExpr(2))));
            launchBound = rewriter.create<AffineApplyOp>(
                loc, annotation.bound().getValue().compose(stepMap),
                ValueRange{
                    ensureLaunchIndependent(
                        cloningMap.lookupOrDefault(upperBound)),
                    ensureLaunchIndependent(
                        cloningMap.lookupOrDefault(lowerBound)),
                    ensureLaunchIndependent(cloningMap.lookupOrDefault(step))});
      // Use the corresponding thread/grid index as replacement for the loop iv.
      Value operand =
          launchOp.body().getArgument(getLaunchOpArgumentNum(processor));
      // Take the indexmap and add the lower bound and step computations in.
      // This computes operand * step + lowerBound.
      // Use an affine map here so that it composes nicely with the provided
      // annotation.
      AffineMap lowerAndStep = AffineMap::get(
          1, 2,
          rewriter.getAffineDimExpr(0) * rewriter.getAffineSymbolExpr(0) +
              rewriter.getAffineSymbolExpr(1));
      newIndex = rewriter.create<AffineApplyOp>(
          loc, annotation.map().getValue().compose(lowerAndStep),
          ValueRange{operand, step, lowerBound});
          }
          // todo(herhut,ravishankarm): Update the behavior of setMappingAttr
          // when this condition is relaxed.
          if (bounds.find(processor) != bounds.end()) {
            return parallelOp.emitOpError()
                   << "cannot redefine the bound for processor "
                   << static_cast<int64_t>(processor);
          }
          bounds[processor] = launchBound;
        }
        if (!boundIsPrecise && !toApproximateIterations) {
          // We are using an approximation, create a surrounding conditional.
          Value originalBound = std::get<3>(config);
          CmpIOp pred = rewriter.create<CmpIOp>(
              loc, CmpIPredicate::slt, newIndex,
              cloningMap.lookupOrDefault(originalBound));
          scf::IfOp ifOp = rewriter.create<scf::IfOp>(loc, pred, false);
          rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
          // Put a sentinel into the worklist so we know when to pop out of the
          // if body again. We use the launchOp here, as that cannot be part of
          // the bodies instruction.
          worklist.push_back(launchOp.getOperation());
        }
      }
    } else {
      // Create a sequential for loop.
      auto loopOp = rewriter.create<scf::ForOp>(
          loc, cloningMap.lookupOrDefault(lowerBound),
          cloningMap.lookupOrDefault(upperBound),
          cloningMap.lookupOrDefault(step));
      newIndex = loopOp.getInductionVar();
      rewriter.setInsertionPointToStart(loopOp.getBody());
      // Put a sentinel into the worklist so we know when to pop out of the loop
      // body again. We use the launchOp here, as that cannot be part of the
      // bodies instruction.
      worklist.push_back(launchOp.getOperation());
    }
    cloningMap.map(iv, newIndex);
    loopIVtoIter[iv] = newIndex;
    mappedParallelLoops.push_back(parallelOp);
    mappedloopIVs[iv] = true;
  }

  // Propagate custom user defined optional attributes, that can be used at
  // later stage, such as extension data for GPU kernel dispatch
  for (const auto &namedAttr : parallelOp.getAttrs()) {
    if (namedAttr.first == gpu::getMappingAttrName() ||
        namedAttr.first == ParallelOp::getOperandSegmentSizeAttr())
      continue;
    launchOp.setAttr(namedAttr.first, namedAttr.second);
  }

  Block *body = parallelOp.getBody();
  worklist.reserve(worklist.size() + body->getOperations().size());
  for (Operation &op : llvm::reverse(body->without_terminator()))
    worklist.push_back(&op);
  return success();
}

/// Lower a `scf.parallel` operation into a corresponding `gpu.launch`
/// operation.
///
/// This essentially transforms a loop nest into a corresponding SIMT function.
/// The conversion is driven by mapping annotations on the `scf.parallel`
/// operations. The mapping is provided via a `DictionaryAttribute` named
/// `mapping`, which has three entries:
///  - processor: the hardware id to map to. 0-2 are block dimensions, 3-5 are
///               thread dimensions and 6 is sequential.
///  - map : An affine map that is used to pre-process hardware ids before
///          substitution.
///  - bound : An affine map that is used to compute the bound of the hardware
///            id based on an upper bound of the number of iterations.
/// If the `scf.parallel` contains nested `scf.parallel` operations, those
/// need to be annotated, as well. Structurally, the transformation works by
/// splicing all operations from nested `scf.parallel` operations into a single
/// sequence. Indices mapped to hardware ids are substituted with those ids,
/// wheras sequential mappings result in a sequential for-loop. To have more
/// flexibility when mapping code to hardware ids, the transform supports two
/// affine maps. The first `map` is used to compute the actual index for
/// substitution from the hardware id. The second `bound` is used to compute the
/// launch dimension for the hardware id from the number of iterations the
/// mapped loop is performing. Note that the number of iterations might be
/// imprecise if the corresponding loop-bounds are loop-dependent. In such case,
/// the hardware id might iterate over additional indices. The transformation
/// caters for this by predicating the created sequence of instructions on
/// the actual loop bound. This only works if an static upper bound for the
/// dynamic loop bound can be derived, currently via analyzing `affine.min`
/// operations.
LogicalResult
ParallelToGpuLaunchLowering::matchAndRewrite(ParallelOp parallelOp,
                                             PatternRewriter &rewriter) const {
  // Create a launch operation. We start with bound one for all grid/block
  // sizes. Those will be refined later as we discover them from mappings.
  Location loc = parallelOp.getLoc();
  Value constantOne = rewriter.create<ConstantIndexOp>(parallelOp.getLoc(), 1);
  gpu::LaunchOp launchOp = rewriter.create<gpu::LaunchOp>(
      parallelOp.getLoc(), constantOne, constantOne, constantOne, constantOne,
      constantOne, constantOne);
  rewriter.setInsertionPointToEnd(&launchOp.body().front());
  rewriter.create<gpu::TerminatorOp>(loc);
  rewriter.setInsertionPointToStart(&launchOp.body().front());

  BlockAndValueMapping cloningMap;
  SmallVector<ParallelOp, 6> mappedParallelLoops;
  llvm::DenseMap<Value, bool> mappedloopIVs;
  llvm::DenseMap<gpu::Processor, Value> launchBounds;
  llvm::DenseMap<Value, Value> loopIVtoIter;
  SmallVector<Operation *, 16> worklist;
  if (failed(processParallelLoop(parallelOp, launchOp, cloningMap, worklist,
                                 launchBounds, mappedParallelLoops,
                                 mappedloopIVs, loopIVtoIter, rewriter)))
    return failure();

  // Whether we have seen any side-effects. Reset when leaving an inner scope.
  bool seenSideeffects = false;
  // Whether we have left a nesting scope (and hence are no longer innermost).
  bool leftNestingScope = false;
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    // Now walk over the body and clone it.
    // TODO: This is only correct if there either is no further scf.parallel
    //       nested or this code is side-effect free. Otherwise we might need
    //       predication. We are overly conservative for now and only allow
    //       side-effects in the innermost scope.
    if (auto nestedParallel = dyn_cast<ParallelOp>(op)) {
      // Before entering a nested scope, make sure there have been no
      // sideeffects until now.
      if (seenSideeffects)
        return failure();
      // A nested scf.parallel needs insertion of code to compute indices.
      // Insert that now. This will also update the worklist with the loops
      // body.
      if (failed(processParallelLoop(
              nestedParallel, launchOp, cloningMap, worklist, launchBounds,
              mappedParallelLoops, mappedloopIVs, loopIVtoIter,rewriter)))
        return failure();
    } else if (op == launchOp.getOperation()) {
      // Found our sentinel value. We have finished the operations from one
      // nesting level, pop one level back up.
      auto parent = rewriter.getInsertionPoint()->getParentOp();
      rewriter.setInsertionPointAfter(parent);
      leftNestingScope = true;
      seenSideeffects = false;
    } else {
      if (ForOp forOp = dyn_cast<ForOp>(op)) {
      // There may be a parallel op inside this for. Find the parallel op
      // and process it.
        Block *body = forOp.getBody();
        for (Operation &op : body->without_terminator()) {
          if (ParallelOp nestedParallelOp = dyn_cast<ParallelOp>(op)) {
            if (failed(processParallelLoop(
                    nestedParallelOp, launchOp, cloningMap, worklist, launchBounds,
                    mappedParallelLoops, mappedloopIVs, loopIVtoIter, rewriter)))
              return failure();
          }
        }
      }
      // Otherwise we copy it over.
      Operation *clone = rewriter.clone(*op, cloningMap);
      cloningMap.map(op->getResults(), clone->getResults());
      // Check for side effects.
      // TODO: Handle region side effects properly.
      seenSideeffects |= !MemoryEffectOpInterface::hasNoEffect(clone) ||
                         clone->getNumRegions() != 0;
      // If we are no longer in the innermost scope, sideeffects are disallowed.
      if (seenSideeffects && leftNestingScope)
        return failure();
    }
  }

  // Now that we succeeded creating the launch operation, also update the
  // bounds.
  for (auto bound : launchBounds)
    launchOp.setOperand(getLaunchOpArgumentNum(std::get<0>(bound)),
                        std::get<1>(bound));
  rewriter.eraseOp(parallelOp);
  launchOp.getParentOfType<FuncOp>().dump();
  return success();
}

void mlir::populateParallelLoopToGPUPatterns(OwningRewritePatternList &patterns,
                                             MLIRContext *ctx) {
  patterns.insert<ParallelToGpuLaunchLowering>(ctx);
}
