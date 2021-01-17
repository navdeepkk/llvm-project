//==-- WmmaStoreOptoNVVMLowering.h - GPU MMA storeOp to NVVM Op lowering --==//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the patterns to lower the GPU subgroup MMA_store op to the
// NVVM Dialect.
//
//===----------------------------------------------------------------------===//

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
struct WmmaStoreOptoNVVMLowering
    : public ConvertOpToLLVMPattern<gpu::SubgroupMmaStoreMatrixOp> {
public:
  MLIRContext *context = &this->getTypeConverter()->getContext();

  explicit WmmaStoreOptoNVVMLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<gpu::SubgroupMmaStoreMatrixOp>(typeConverter),
        llvmTypes(context) {}

  static LogicalResult areAllLLVMTypes(Operation *op, ValueRange operands,
                                       ConversionPatternRewriter &rewriter) {
    if (!llvm::all_of(operands, [](Value value) {
          return value.getType().isa<LLVM::LLVMType>();
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

    // Source memref of the original op.
    MemRefType srcMemrefType =
        subgroupMmaStoreMatrixOp.srcMemref().getType().cast<MemRefType>();
    Location loc = op->getLoc();
    ArrayRef<int64_t> srcMemrefShape = srcMemrefType.getShape();

    // Destination memref of the original op.
    MemRefType dstMemrefType =
        subgroupMmaStoreMatrixOp.dstMemref().getType().cast<MemRefType>();

    // Promote operands of this op. This emits !llvm.extractelement for each
    // of the operand memrefs and makes it easy to use these values in
    // subsequent instruction.
    auto promotedSrcOp = this->getTypeConverter()->promoteOperands(
        loc, op->getOperand(0), operands[0], rewriter);

    auto promotedDstOp = this->getTypeConverter()->promoteOperands(
        loc, op->getOperand(1), operands[1],
        rewriter);

    auto dstOffsetI = subgroupMmaStoreMatrixOp.dstOffsetIAttr();
    auto dstOffsetJ = subgroupMmaStoreMatrixOp.dstOffsetJAttr();
    auto wm = subgroupMmaStoreMatrixOp.wmAttr();
    auto wn = subgroupMmaStoreMatrixOp.wnAttr();
    auto wk = subgroupMmaStoreMatrixOp.wkAttr();
    auto ldm = subgroupMmaStoreMatrixOp.ldmAttr();

    // Emit ops which compute the store offset using `dstOffsetI`,
    // `dstOffsetJ`. The actualOffset is (memrefOffset + (alignedPtr + ((ldm *
    // dstOffsetI) + dstOffsetJ)).
    Value dstOffsetIVal = rewriter.create<LLVM::ConstantOp>(
        loc, llvmTypes.llvmInt64Type, dstOffsetI);
    Value dstOffsetJVal = rewriter.create<LLVM::ConstantOp>(
        loc, llvmTypes.llvmInt64Type, dstOffsetJ);
    Value leadingDim64 =
        rewriter.create<LLVM::ConstantOp>(loc, llvmTypes.llvmInt64Type, ldm);
    Value numElemsLeadDim = rewriter.create<LLVM::MulOp>(
        loc, llvmTypes.llvmInt64Type, leadingDim64, dstOffsetIVal);
    Value loadOffset = rewriter.create<LLVM::AddOp>(
        loc, llvmTypes.llvmInt64Type, numElemsLeadDim, dstOffsetJVal);
    // Cast offset I64 to make the calculation below independent of index
    // bitwidth supplied.
    Value promotedDstOpToUse;
    int64_t indexTypeBitwidth =
        this->getTypeConverter()->getIndexTypeBitwidth();
    if (indexTypeBitwidth < 64)
      promotedDstOpToUse = rewriter.create<LLVM::SExtOp>(
          loc, llvmTypes.llvmInt64Type, promotedDstOp[2]);
    else if (indexTypeBitwidth > 64)
      promotedDstOpToUse = rewriter.create<LLVM::TruncOp>(
          loc, llvmTypes.llvmInt64Type, promotedDstOp[2]);
    else
      promotedDstOpToUse = promotedDstOp[2];
    Value actualOffset = rewriter.create<LLVM::AddOp>(
        loc, llvmTypes.llvmInt64Type, loadOffset, promotedDstOpToUse);
    Value storeAddress = rewriter.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(llvmTypes.llvmF16Ty,
                                   dstMemrefType.getMemorySpace()),
        promotedDstOp[1], ArrayRef<Value>{actualOffset});

    // Bitcast the base address pointer of the destination memref, So that
    // values can be stored in chunks of 32-bits, as they were returned by the
    // wmmaLoadOP.
    Value storeAddressCasted = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(llvmTypes.llvmInt32Type,
                                   dstMemrefType.getMemorySpace()),
        storeAddress);

    SmallVector<Value, 4> storeOpOperands;
    storeOpOperands.push_back(storeAddressCasted);

    // Unpack the results from the source memref.
    for (unsigned i = 0, e = srcMemrefShape[0]; i < e; ++i) {
      Value loadAddress = rewriter.create<LLVM::GEPOp>(
          loc,
          LLVM::LLVMPointerType::get(llvmTypes.llvmF16x2Ty,
                                     /*NVVM private memory space*/ 0),
          promotedSrcOp[1],
          ArrayRef<Value>{rewriter.create<LLVM::ConstantOp>(
              loc, llvmTypes.llvmInt32Type, rewriter.getUI32IntegerAttr(i))});

      storeOpOperands.push_back(
          rewriter.create<LLVM::LoadOp>(loc, loadAddress));
    }

    // For NVPTX intrinsic compatibility, create an I32 constant op for ldm.
    // This might result in loss of data. leadingDim is in number of elements
    // as required by the NVPTX instrinsic.
    Value leadingDim32 =
        rewriter.create<LLVM::ConstantOp>(loc, llvmTypes.llvmInt32Type, ldm);
    storeOpOperands.push_back(leadingDim32);

    // Create nvvm.mma_store op.
    ValueRange unpackedValueRange(storeOpOperands);
    rewriter.create<NVVM::WMMAStoreOp>(
        loc, storeOpOperands, wm.cast<mlir::IntegerAttr>(),
        wn.cast<mlir::IntegerAttr>(), wk.cast<mlir::IntegerAttr>(),
        ldm.cast<mlir::IntegerAttr>());

    rewriter.eraseOp(op);
    return success();
  }

private:
  CommonLLVMTypes llvmTypes;
};
} // namespace mlir
