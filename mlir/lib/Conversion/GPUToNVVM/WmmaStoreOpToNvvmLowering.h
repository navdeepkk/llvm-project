//==- WmmaStoreOpToNVVMLoering.h - MmaStoreOp to NVVM Op lowering *- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains patterns to lower the GPU subgroup MMA_store op to the
// NVVM Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_CONVERSION_GPUTONVVM_WMMAMMAOPTONVVMLOWERING_H
#define MLIR_LIB_CONVERSION_GPUTONVVM_WMMAMMAOPTONVVMLOWERING_H

#include "CommonTypes.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

namespace mlir {
/// This class implemtents the conversion of GPU MMA storeOp to wmma.store op
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

    // Unpack the results from the source memref.
    if (subgroupMmaStoreMatrixOp.src()
            .getType()
            .cast<gpu::MMAFragmentType>()
            .getElementType() == llvmTypes.f16x2Ty) {
      for (unsigned i = 0, e = llvmTypes.numHalfsInOpFrags[llvmTypes.D]; i < e;
           ++i) {
        Value toUse = rewriter.create<LLVM::ExtractValueOp>(
            loc, llvmTypes.f16x2Ty, operands[0], rewriter.getI64ArrayAttr(i));
        storeOpOperands.push_back(toUse);
      }
      storeOpOperands.push_back(leadingDim32);

      // Create nvvm.mma_store op.
      ValueRange unpackedValueRange(storeOpOperands);
      rewriter.create<NVVM::WMMAStoreF16Op>(loc, storeOpOperands);

      rewriter.eraseOp(op);
      return success();
    } else if (subgroupMmaStoreMatrixOp.src()
                   .getType()
                   .cast<gpu::MMAFragmentType>()
                   .getElementType() == llvmTypes.f32Ty) {
      for (unsigned i = 0, e = 8; i < e; ++i) {
        Value toUse = rewriter.create<LLVM::ExtractValueOp>(
            loc, llvmTypes.f32Ty, operands[0], rewriter.getI64ArrayAttr(i));
        storeOpOperands.push_back(toUse);
      }
      storeOpOperands.push_back(leadingDim32);

      // Create nvvm.mma_store op.
      ValueRange unpackedValueRange(storeOpOperands);
      rewriter.create<NVVM::WMMAStoreF32Op>(loc, storeOpOperands);

      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }

private:
  /// Contains definitions of all the LLVM types which are used for lowering
  /// this GPU SubgroupMmaStoreMatrixOp.
  CommonLLVMTypes llvmTypes;
};
} // namespace mlir

#endif
