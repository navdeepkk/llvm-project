//===-- WmmaLoadOpToNVVMLowering.h - MmaLoadOp to NVVM lowering -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains patterns to lower the GPU subgroup MMA_loadOp to NVVM
// Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_CONVERSION_GPUTONVVM_WMMALOADOPTONVVMLOWERING_H
#define MLIR_LIB_CONVERSION_GPUTONVVM_WMMALOADOPTONVVMLOWERING_H

#include "CommonTypes.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

namespace mlir {

static LogicalResult areAllLLVMTypes(Operation *op, ValueRange operands,
                                     ConversionPatternRewriter &rewriter) {
  if (!llvm::all_of(operands, [](Value value) {
        return LLVM::isCompatibleType(value.getType());
      }))
    return rewriter.notifyMatchFailure(
        op, "Cannot convert if operands aren't of LLVM type.");

  return success();
}

template <typename T>
static void linearizeIndices(T &loadStoreOp, MemRefType memRefType,
                             ConversionPatternRewriter &rewriter, Type opType,
                             Value &res) {
  Location loc = loadStoreOp.getLoc();
  SmallVector<Value> numElems;

  // Create a constant 1.
  numElems.push_back(rewriter.create<LLVM::ConstantOp>(
      loc, opType, rewriter.getI32IntegerAttr(1)));

  // Accumulate the dim count.
  for (int64_t dimInx = memRefType.getShape().size() - 1, e = 0; dimInx > e;
       dimInx--) {
    Value dim = (rewriter.create<LLVM::ConstantOp>(
        loc, opType,
        rewriter.getI32IntegerAttr(memRefType.getDimSize(dimInx))));
    numElems.push_back(
        rewriter.create<LLVM::MulOp>(loc, opType, dim, numElems.back()));
  }

  // Reverse the list for easy manipulation.
  std::reverse(numElems.begin(), numElems.end());

  // Create a linear index by multiplying the dim count with the respective
  // index.
  SmallVector<Value> linearIdx;
  linearIdx.push_back(rewriter.create<LLVM::MulOp>(
      loc, opType, loadStoreOp.indices()[0], numElems[0]));

  for (uint64_t i = 1; i < loadStoreOp.indices().size(); ++i) {
    Value prod = rewriter.create<LLVM::MulOp>(
        loc, opType, loadStoreOp.indices()[i], numElems[i]);
    linearIdx.push_back(
        rewriter.create<LLVM::AddOp>(loc, opType, linearIdx[i - 1], prod));
  }

  res = linearIdx.back();
}

/// This class implemtents the conversion of GPU MMA loadOp to wmma.load op
/// in the NVVM dialect. The conversion not only emits the NVVM op but also
/// emits code that is necessary to store the data in the destination memref
/// after it has been loaded.
struct WmmaLoadOpToNVVMLowering
    : public ConvertOpToLLVMPattern<gpu::SubgroupMmaLoadMatrixOp> {
public:
  MLIRContext *context = &this->getTypeConverter()->getContext();

  explicit WmmaLoadOpToNVVMLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<gpu::SubgroupMmaLoadMatrixOp>(typeConverter),
        llvmTypes(context) {}

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaLoadMatrixOp subgroupMmaLoadMatrixOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *op = subgroupMmaLoadMatrixOp.getOperation();
    if (failed(areAllLLVMTypes(op, operands, rewriter)))
      return failure();

    int8_t indexTypeBitwidth = this->getTypeConverter()->getIndexTypeBitwidth();

    // The corresponding intrinsics expects leadDimension to be a 32-bit
    // integer, so all the calculations of linearizing the load address
    // must also follow this restriction.
    if (indexTypeBitwidth != 32)
      return rewriter.notifyMatchFailure(
          op, "Expected indices to the memref to be 32-bit wide.");

    // Source memref of the original op.
    MemRefType srcMemrefType =
        subgroupMmaLoadMatrixOp.srcMemref().getType().cast<MemRefType>();
    Location loc = op->getLoc();

    auto leadDimension = subgroupMmaLoadMatrixOp.leadDimensionAttr();
    auto operand = subgroupMmaLoadMatrixOp.operandAttr();

    // Emit information for the memref operands.
    auto promotedSrcOp = this->getTypeConverter()->promoteOperands(
        loc, op->getOperand(0), operands[0], rewriter);

    // Emit ops which compute the load offset using `srcOffsetI`,
    // `srcOffsetJ`. The actualOffset is (memrefOffset + (alignedPtr +
    // ((leadDimension * srcOffsetI) + srcOffsetJ)). The memrefs here are
    // assumed to be normalized and hence the simple conversion works.
    Value loadOffset;
    linearizeIndices(subgroupMmaLoadMatrixOp, srcMemrefType, rewriter,
                     llvmTypes.int32Type, loadOffset);
    // Value srcOffsetIVal =
    // subgroupMmaLoadMatrixOp->getOpOperand(beginInx).get(); Value
    // srcOffsetJVal
    // =
    //    subgroupMmaLoadMatrixOp->getOpOperand(beginInx + 1).get();
    Value leadingDim32 = rewriter.create<LLVM::ConstantOp>(
        loc, llvmTypes.int32Type, leadDimension);
    // Value numElemsLeadDim = rewriter.create<LLVM::MulOp>(
    //    loc, llvmTypes.int32Type, leadingDim32, srcOffsetIVal);
    // Value loadOffset = rewriter.create<LLVM::AddOp>(
    //    loc, llvmTypes.int32Type, numElemsLeadDim, srcOffsetJVal);
    Value promotedSrcOpToUse;

    promotedSrcOpToUse = promotedSrcOp[2];
    Value actualOffset = rewriter.create<LLVM::AddOp>(
        loc, llvmTypes.int32Type, loadOffset, promotedSrcOpToUse);
    Value loadAddress = rewriter.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(llvmTypes.f16Ty,
                                   srcMemrefType.getMemorySpaceAsInt()),
        promotedSrcOp[1], ArrayRef<Value>{actualOffset});

    // Bitcast the pointer from *half to *i32 so that it matches the semantics
    // of the inrinsic exposed by the NVPTX backend.
    Value loadAddressCasted = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(llvmTypes.int32Type,
                                   srcMemrefType.getMemorySpaceAsInt()),
        loadAddress);

    Type resType;
    StringRef operandStr = operand.cast<mlir::StringAttr>().getValue();

    if (operandStr.equals("AOp") || operandStr.equals("BOp")) {
      resType = llvmTypes.fragArrayABTy;
    } else {
      if (srcMemrefType.getElementType().isF16())
        resType = llvmTypes.fragArrayCDTy;
      else if (srcMemrefType.getElementType().isF32())
        resType = llvmTypes.fragArrayCDF32Ty;
      else
        return failure();
    }

    SmallVector<Value, 2> loadOpOperands({loadAddressCasted, leadingDim32});

    // Create nvvm.mma_load op according to the operand types.
    if (operandStr.equals("AOp")) {
      NVVM::WMMALoadAOp wmmaLoadAOp =
          rewriter.create<NVVM::WMMALoadAOp>(loc, resType, loadOpOperands);
      rewriter.replaceOp(op, wmmaLoadAOp.getResult());
    } else if (operandStr.equals("BOp")) {
      NVVM::WMMALoadBOp wmmaLoadBOp =
          rewriter.create<NVVM::WMMALoadBOp>(loc, resType, loadOpOperands);
      rewriter.replaceOp(op, wmmaLoadBOp.getResult());
    } else {
      if (srcMemrefType.getElementType().isF16()) {
        NVVM::WMMALoadCF16Op wmmaLoadCOp =
            rewriter.create<NVVM::WMMALoadCF16Op>(loc, resType, loadOpOperands);
        rewriter.replaceOp(op, wmmaLoadCOp.getResult());
      } else if (srcMemrefType.getElementType().isF32()) {
        NVVM::WMMALoadCF32Op wmmaLoadCOp =
            rewriter.create<NVVM::WMMALoadCF32Op>(loc, resType, loadOpOperands);
        rewriter.replaceOp(op, wmmaLoadCOp.getResult());
      }
    }

    return success();
  }

private:
  /// Contains definitions of all the LLVM types which are used for lowering
  /// this GPU subgroupMmaLoadMatrixOp.
  CommonLLVMTypes llvmTypes;
};

/// This class implements the conversion of GPU MMA storeOp to wmma.store op
/// in the NVVM dialect. The conversion not only emits the NVVM op but also
/// emits code that is necessary to unpack the data in the source memref and
/// convert the data in the format that is needed by the NVVM op.
struct WmmaStoreOpToNVVMLowering
    : public ConvertOpToLLVMPattern<gpu::SubgroupMmaStoreMatrixOp> {
public:
  MLIRContext *context = &this->getTypeConverter()->getContext();

  explicit WmmaStoreOpToNVVMLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<gpu::SubgroupMmaStoreMatrixOp>(typeConverter),
        llvmTypes(context) {}

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaStoreMatrixOp subgroupMmaStoreMatrixOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *op = subgroupMmaStoreMatrixOp.getOperation();
    if (failed(areAllLLVMTypes(op, operands, rewriter)))
      return failure();

    int8_t indexTypeBitwidth = this->getTypeConverter()->getIndexTypeBitwidth();
    // The corresponding intrinsics expects leadDimension to be a 32-bit
    // integer, so all the calculations of linearizing the load address
    // must also follow this restriction.
    if (indexTypeBitwidth != 32)
      return rewriter.notifyMatchFailure(
          op, "Expected indices to the meref to be 32-bit wide.");

    Location loc = op->getLoc();

    // Destination memref of the original op.
    MemRefType dstMemrefType =
        subgroupMmaStoreMatrixOp.dstMemref().getType().cast<MemRefType>();

    auto promotedDstOp = this->getTypeConverter()->promoteOperands(
        loc, op->getOperand(1), operands[1], rewriter);

    auto leadDimension = subgroupMmaStoreMatrixOp.leadDimensionAttr();
    Value loadOffset;
    linearizeIndices(subgroupMmaStoreMatrixOp, dstMemrefType, rewriter,
                     llvmTypes.int32Type, loadOffset);
    // Emit ops which compute the store offset using `dstOffsetI`,
    // `dstOffsetJ`. The actualOffset is (memrefOffset + (alignedPtr +
    // ((leadDimension * dstOffsetI) + dstOffsetJ)).
    // Value dstOffsetIVal = subgroupMmaStoreMatrixOp.getOperand(beginInx);
    // Value dstOffsetJVal = subgroupMmaStoreMatrixOp.getOperand(beginInx + 1);
    Value leadingDim32 = rewriter.create<LLVM::ConstantOp>(
        loc, llvmTypes.int32Type, leadDimension);
    // Value numElemsLeadDim = rewriter.create<LLVM::MulOp>(
    //    loc, llvmTypes.int32Type, leadingDim32, dstOffsetIVal);
    // Value loadOffset = rewriter.create<LLVM::AddOp>(
    //    loc, llvmTypes.int32Type, numElemsLeadDim, dstOffsetJVal);
    // Cast offset I64 to make the calculation below independent of index
    // bitwidth supplied.
    Value promotedDstOpToUse;

    promotedDstOpToUse = promotedDstOp[2];
    Value actualOffset = rewriter.create<LLVM::AddOp>(
        loc, llvmTypes.int32Type, loadOffset, promotedDstOpToUse);
    Value storeAddress = rewriter.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(llvmTypes.f16Ty,
                                   dstMemrefType.getMemorySpaceAsInt()),
        promotedDstOp[1], ArrayRef<Value>{actualOffset});

    // Bitcast the base address pointer of the destination memref, So that
    // values can be stored in chunks of 32-bits.
    Value storeAddressCasted = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(llvmTypes.int32Type,
                                   dstMemrefType.getMemorySpaceAsInt()),
        storeAddress);

    SmallVector<Value, 4> storeOpOperands;
    storeOpOperands.push_back(storeAddressCasted);

    // Unpack the results from the source.
    if (subgroupMmaStoreMatrixOp.src()
            .getType()
            .cast<gpu::MMAFragmentType>()
            .getElementType() == llvmTypes.f16x2Ty) {
      for (unsigned i = 0, e = llvmTypes.numHalfsInOpFrags[llvmTypes.D]; i < e;
           ++i) {
        Value toUse = rewriter.create<LLVM::ExtractValueOp>(
            loc, llvmTypes.f16x2Ty, operands[0], rewriter.getI32ArrayAttr(i));
        storeOpOperands.push_back(toUse);
      }
      storeOpOperands.push_back(leadingDim32);

      // Create nvvm.mma_store op.
      rewriter.create<NVVM::WMMAStoreF16Op>(loc, storeOpOperands);

      rewriter.eraseOp(op);
      return success();
    }
    if (subgroupMmaStoreMatrixOp.src()
            .getType()
            .cast<gpu::MMAFragmentType>()
            .getElementType() == llvmTypes.f32Ty) {
      for (unsigned i = 0, e = 8; i < e; ++i) {
        Value toUse = rewriter.create<LLVM::ExtractValueOp>(
            loc, llvmTypes.f32Ty, operands[0], rewriter.getI32ArrayAttr(i));
        storeOpOperands.push_back(toUse);
      }
      storeOpOperands.push_back(leadingDim32);

      // Create nvvm.mma_store op.
      rewriter.create<NVVM::WMMAStoreF32Op>(loc, storeOpOperands);

      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }

private:
  /// Definitions of all the LLVM types which are used for lowering this GPU
  /// SubgroupMmaStoreMatrixOp.
  CommonLLVMTypes llvmTypes;
};
} // namespace mlir

#endif
