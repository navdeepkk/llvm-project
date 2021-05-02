//===- WmmaLoadStoreOpToNVVMLowering.h - LD/ST to NVVM lowering -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains patterns to lower the GPU subgroup MMA_loadOp and
// MMA_storeop to NVVM Dialect.
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
          op, "Expected indices to the meref to be 32-bit wide.");

    // Source memref of the original op.
    MemRefType srcMemrefType =
        subgroupMmaLoadMatrixOp.srcMemref().getType().cast<MemRefType>();
    Location loc = op->getLoc();

    auto beginInx = subgroupMmaLoadMatrixOp.indices().getBeginOperandIndex();
    auto leadDimension = subgroupMmaLoadMatrixOp.leadDimensionAttr();
    // auto operand = subgroupMmaLoadMatrixOp.operandAttr();

    // Emit information for the memref operands.
    auto promotedSrcOp = this->getTypeConverter()->promoteOperands(
        loc, op->getOperand(0), operands[0], rewriter);

    // Emit ops which compute the load offset using `srcOffsetI`,
    // `srcOffsetJ`. The actualOffset is (memrefOffset + (alignedPtr +
    // ((leadDimension * srcOffsetI) + srcOffsetJ)). The memrefs here are
    // assumed to be normalized and hence the simple conversion works.
    Value srcOffsetIVal = subgroupMmaLoadMatrixOp->getOpOperand(beginInx).get();
    Value srcOffsetJVal =
        subgroupMmaLoadMatrixOp->getOpOperand(beginInx + 1).get();
    Value leadingDim32 = rewriter.create<LLVM::ConstantOp>(
        loc, llvmTypes.int32Type, leadDimension);
    Value numElemsLeadDim = rewriter.create<LLVM::MulOp>(
        loc, llvmTypes.int32Type, leadingDim32, srcOffsetIVal);
    Value loadOffset = rewriter.create<LLVM::AddOp>(
        loc, llvmTypes.int32Type, numElemsLeadDim, srcOffsetJVal);

    Value promotedSrcOpToUse;
    promotedSrcOpToUse = promotedSrcOp[2];
    Value actualOffset = rewriter.create<LLVM::AddOp>(
        loc, llvmTypes.int32Type, loadOffset, promotedSrcOpToUse);
    Value loadAddress = rewriter.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(llvmTypes.f16Ty,
                                   srcMemrefType.getMemorySpaceAsInt()),
        promotedSrcOp[1], ArrayRef<Value>{actualOffset});

    // Bitcast the base address pointer of the destination memref, So that
    // values can be stored in chunks of 32-bits and semantics mathc with the
    // intrinsic exposed by NVPTX backend.
    Value loadAddressCasted = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(llvmTypes.int32Type,
                                   srcMemrefType.getMemorySpaceAsInt()),
        loadAddress);

    // Get the shape of the MMAMatrix type being returned. The shape will
    // choose which intrinsic this op will be lowered to.
    gpu::MMAMatrixType retType =
        subgroupMmaLoadMatrixOp.res().getType().cast<gpu::MMAMatrixType>();
    ArrayRef<int64_t> retTypeShape = retType.getShape();

    Type resType;
    // StringRef operandStr = operand.cast<mlir::StringAttr>().getValue();
    StringRef operandStr = retType.getOperand();
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

    // Create nvvm.mma_load op according to the operand types.
    SmallVector<Value, 2> loadOpOperands({loadAddressCasted, leadingDim32});
    if (operandStr.equals("AOp")) {
      if (retTypeShape[0] == 16 && retTypeShape[1] == 16) {
        NVVM::WMMALoadAM16N16K16Op wmmaLoadAOp =
            rewriter.create<NVVM::WMMALoadAM16N16K16Op>(loc, resType,
                                                        loadOpOperands);
        rewriter.replaceOp(op, wmmaLoadAOp.getResult());
      } else {
        // Only M16N16K16 version implemented. Add cases as more versions are
        // implemented.
        return failure();
      }
    } else if (operandStr.equals("BOp")) {
      if (retTypeShape[0] == 16 && retTypeShape[1] == 16) {
        NVVM::WMMALoadBM16N16K16Op wmmaLoadBOp =
            rewriter.create<NVVM::WMMALoadBM16N16K16Op>(loc, resType,
                                                        loadOpOperands);
        rewriter.replaceOp(op, wmmaLoadBOp.getResult());
      } else {
        // Only M16N16K16 version implemented. Add cases as more versions are
        // implemented.
        return failure();
      }
    } else {
      if (retTypeShape[0] == 16 && retTypeShape[1] == 16) {
        if (srcMemrefType.getElementType().isF16()) {
          NVVM::WMMALoadCF16M16N16K16Op wmmaLoadCOp =
              rewriter.create<NVVM::WMMALoadCF16M16N16K16Op>(loc, resType,
                                                             loadOpOperands);
          rewriter.replaceOp(op, wmmaLoadCOp.getResult());
        } else if (srcMemrefType.getElementType().isF32()) {
          NVVM::WMMALoadCF32M16N16K16Op wmmaLoadCOp =
              rewriter.create<NVVM::WMMALoadCF32M16N16K16Op>(loc, resType,
                                                             loadOpOperands);
          rewriter.replaceOp(op, wmmaLoadCOp.getResult());
        }
      } else {
        // Only M16N16K16 version implemented. Add cases as more versions are
        // implemented.
        return failure();
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
/// emits code that is necessary to unpack the data in the source and convert
/// the data in the format that is needed by the NVVM op.
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
    unsigned beginInx =
        subgroupMmaStoreMatrixOp.indices().getBeginOperandIndex();

    // Emit ops which compute the store offset using `dstOffsetI`,
    // `dstOffsetJ`. The actualOffset is (memrefOffset + (alignedPtr +
    // ((leadDimension * dstOffsetI) + dstOffsetJ)).
    Value dstOffsetIVal = subgroupMmaStoreMatrixOp.getOperand(beginInx);
    Value dstOffsetJVal = subgroupMmaStoreMatrixOp.getOperand(beginInx + 1);
    Value leadingDim32 = rewriter.create<LLVM::ConstantOp>(
        loc, llvmTypes.int32Type, leadDimension);
    Value numElemsLeadDim = rewriter.create<LLVM::MulOp>(
        loc, llvmTypes.int32Type, leadingDim32, dstOffsetIVal);
    Value loadOffset = rewriter.create<LLVM::AddOp>(
        loc, llvmTypes.int32Type, numElemsLeadDim, dstOffsetJVal);

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
    // values can be stored in chunks of 32-bits and semantics mathc with the
    // intrinsic exposed by NVPTX backend.
    Value storeAddressCasted = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(llvmTypes.int32Type,
                                   dstMemrefType.getMemorySpaceAsInt()),
        storeAddress);

    SmallVector<Value, 4> storeOpOperands;
    storeOpOperands.push_back(storeAddressCasted);

    // Get the shape of the MMAMatrix type being stored. The shape will
    // choose which intrinsic this op will be lowered to.
    gpu::MMAMatrixType srcType =
        subgroupMmaStoreMatrixOp.src().getType().cast<gpu::MMAMatrixType>();
    ArrayRef<int64_t> srcTypeShape = srcType.getShape();

    // Unpack the results from the source.
    if (subgroupMmaStoreMatrixOp.src()
            .getType()
            .cast<gpu::MMAMatrixType>()
            .getElementType() == llvmTypes.f16Ty) {
      for (unsigned i = 0, e = llvmTypes.numHalfsInOpFrags[llvmTypes.D]; i < e;
           ++i) {
        Value toUse = rewriter.create<LLVM::ExtractValueOp>(
            loc, llvmTypes.f16x2Ty, operands[0], rewriter.getI32ArrayAttr(i));
        storeOpOperands.push_back(toUse);
      }
      storeOpOperands.push_back(leadingDim32);

      // Create nvvm.mma_store op.
      if (srcTypeShape[0] == 16 && srcTypeShape[1] == 16)
        rewriter.create<NVVM::WMMAStoreF16M16N16K16Op>(loc, storeOpOperands);
      else
        // Only M16N16K16 version implemented. Add cases as more versions are
        // implemented.
        return failure();

      rewriter.eraseOp(op);
      return success();
    }
    if (subgroupMmaStoreMatrixOp.src()
            .getType()
            .cast<gpu::MMAMatrixType>()
            .getElementType() == llvmTypes.f32Ty) {
      for (unsigned i = 0, e = 8; i < e; ++i) {
        Value toUse = rewriter.create<LLVM::ExtractValueOp>(
            loc, llvmTypes.f32Ty, operands[0], rewriter.getI32ArrayAttr(i));
        storeOpOperands.push_back(toUse);
      }
      storeOpOperands.push_back(leadingDim32);

      // Create nvvm.mma_store op.
      if (srcTypeShape[0] == 16 && srcTypeShape[1] == 16)
        rewriter.create<NVVM::WMMAStoreF32M16N16K16Op>(loc, storeOpOperands);
      else
        // Only M16N16K16 version implemented. Add cases as more versions are
        // implemented.
        return failure();

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
