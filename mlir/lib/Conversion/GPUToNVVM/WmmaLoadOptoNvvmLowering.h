//===-- WmmaLoadOptoNVVMLowering.h - MmaLoadOp to NVVM lowering -*- C++ -*-===//
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
/// This class implemtents the conversion of GPU MMA loadOp to wmma.load op
/// in the NVVM dialect. The conversion not only emits the NVVM op but also
/// emits code that is necessary to store the data in the destination memref
/// after it has been loaded.
struct WmmaLoadOptoNVVMLowering
    : public ConvertOpToLLVMPattern<gpu::SubgroupMmaLoadMatrixOp> {
public:
  MLIRContext *context = &this->getTypeConverter()->getContext();

  explicit WmmaLoadOptoNVVMLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<gpu::SubgroupMmaLoadMatrixOp>(typeConverter),
        llvmTypes(context) {}

  static LogicalResult areAllLLVMTypes(Operation *op, ValueRange operands,
                                       ConversionPatternRewriter &rewriter) {
    if (!llvm::all_of(operands, [](Value value) {
          return LLVM::isCompatibleType(value.getType());
        }))
      return rewriter.notifyMatchFailure(
          op, "Cannot convert if operands aren't of LLVM type.");

    return success();
  }

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
    auto operand = subgroupMmaLoadMatrixOp.operandAttr();

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
    // Cast offset I64 to make the calculation below independent of index
    // bitwidth supplied.
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
      else if(srcMemrefType.getElementType().isF32())
	resType = llvmTypes.fragArrayCDF32Ty;
    }

    ValueRange loadOpOperands({loadAddressCasted, leadingDim32});

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
} // namespace mlir

#endif
