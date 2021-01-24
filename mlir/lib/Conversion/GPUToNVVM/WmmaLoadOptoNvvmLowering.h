//===-- WmmaLoadOptoNVVMLowering.h - GPU MMA loadOp to NVVM Op lowering ---===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the patterns to lower the GPU subgroup MMA loadOp to NVVM
// Dialect.
//
//===----------------------------------------------------------------------===//

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
          return value.getType().isa<LLVM::LLVMType>();
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

    // Source memref of the original op.
    MemRefType srcMemrefType =
        subgroupMmaLoadMatrixOp.srcMemref().getType().cast<MemRefType>();
    Location loc = op->getLoc();

    auto beginInx = subgroupMmaLoadMatrixOp.indices().getBeginOperandIndex();
    auto ldm = subgroupMmaLoadMatrixOp.ldmAttr();
    auto operand = subgroupMmaLoadMatrixOp.operandAttr();

    // Emit information for the memref operands.
    auto promotedSrcOp = this->getTypeConverter()->promoteOperands(
        loc, op->getOperand(0), operands[0], rewriter);

    auto promotedDstOp = this->getTypeConverter()->promoteOperands(
        loc, op->getOperand(1), operands[1], rewriter);

    // Emit ops which compute the load offset using `srcOffsetI`,
    // `srcOffsetJ`. The actualOffset is (memrefOffset + (alignedPtr + ((ldm *
    // srcOffsetI) + srcOffsetJ)). The memrefs here are assumed to be normalized
    // and hence the simple conversion works.
    Value srcOffsetIVal = subgroupMmaLoadMatrixOp->getOpOperand(beginInx).get();
    Value srcOffsetJVal =
        subgroupMmaLoadMatrixOp->getOpOperand(beginInx + 1).get();
    Value leadingDim64 =
        rewriter.create<LLVM::ConstantOp>(loc, llvmTypes.llvmInt64Type, ldm);
    Value numElemsLeadDim = rewriter.create<LLVM::MulOp>(
        loc, llvmTypes.llvmInt64Type, leadingDim64, srcOffsetIVal);
    Value loadOffset = rewriter.create<LLVM::AddOp>(
        loc, llvmTypes.llvmInt64Type, numElemsLeadDim, srcOffsetJVal);
    // Cast offset I64 to make the calculation below independent of index
    // bitwidth supplied.
    Value promotedSrcOpToUse;
    int64_t indexTypeBitwidth =
        this->getTypeConverter()->getIndexTypeBitwidth();
    if (indexTypeBitwidth < 64)
      promotedSrcOpToUse = rewriter.create<LLVM::SExtOp>(
          loc, llvmTypes.llvmInt64Type, promotedSrcOp[2]);
    else if (indexTypeBitwidth > 64)
      promotedSrcOpToUse = rewriter.create<LLVM::TruncOp>(
          loc, llvmTypes.llvmInt64Type, promotedSrcOp[2]);
    else
      promotedSrcOpToUse = promotedSrcOp[2];
    Value actualOffset = rewriter.create<LLVM::AddOp>(
        loc, llvmTypes.llvmInt64Type, loadOffset, promotedSrcOpToUse);
    Value loadAddress = rewriter.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(llvmTypes.llvmF16Ty,
                                   srcMemrefType.getMemorySpace()),
        promotedSrcOp[1], ArrayRef<Value>{actualOffset});

    // Bitcast the pointer from *half to *i32 so that it matches the semantics
    // of the inrinsic exposed by the NVPTX backend.
    Value loadAddressCasted = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(llvmTypes.llvmInt32Type,
                                   srcMemrefType.getMemorySpace()),
        loadAddress);

    // Result types for wmmaLoadOp.
    LLVM::LLVMType resType, dstMemrefElemType;
    unsigned numElemsInResFrag;
    if (operand.cast<mlir::StringAttr>().getValue().equals("AOp") ||
        operand.cast<mlir::StringAttr>().getValue().equals("BOp")) {
      resType = llvmTypes.fragArrayABTy;
      dstMemrefElemType = llvmTypes.llvmF16x16Ty;
      numElemsInResFrag = llvmTypes.numHalfsInOpFrags[llvmTypes.A];
    } else {
      resType = llvmTypes.fragArrayCDTy;
      dstMemrefElemType = llvmTypes.llvmF16x8Ty;
      numElemsInResFrag = llvmTypes.numHalfsInOpFrags[llvmTypes.C];
    }

    // For NVPTX intrinsic compatibility, create an I32 constant op for ldm.
    // This might result in loss of data. leadingDim is in number of elements
    // as required by the NVPTX instrinsic.
    Value leadingDim32 =
        rewriter.create<LLVM::ConstantOp>(loc, llvmTypes.llvmInt32Type, ldm);

    // Create nvvm.mma_load op according to the operand.
    ValueRange loadOpOperands({loadAddressCasted, leadingDim32});

    NVVM::WMMALoadOp wmmaLoadOp = rewriter.create<NVVM::WMMALoadOp>(
        loc, resType, loadOpOperands, operand.cast<mlir::StringAttr>());

    // Get the store address for this fragment in the destination memref.
    Value dstStoreAddress = rewriter.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(dstMemrefElemType,
                                   /*NVVM private memory space*/ 0),
        promotedDstOp[1], subgroupMmaLoadMatrixOp.dstIndex());

    // Bitcast the base address pointer of the destination memref, So that
    // values can be stored in chunks of 32-bits, as they were returned by the
    // wmmaLoadOP.
    Value storeAddressCasted = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(llvmTypes.llvmInt32Type,
                                   /*NVVM private memory space*/ 0),
        dstStoreAddress);

    // Move the data into the memref that was passed as an argument to the
    // original op. The result of the op is a !llvm.struct, the results are to
    // be moved into the memref element by element. The number of elements in
    // the memref and the number of elements in the struct should be same.
    for (unsigned i = 0, e = numElemsInResFrag; i < e; ++i) {
      Value toStore = rewriter.create<LLVM::ExtractValueOp>(
          loc, llvmTypes.llvmF16x2Ty, wmmaLoadOp,
          rewriter.getIndexArrayAttr(i));
      Value toStoreI32 = rewriter.create<LLVM::BitcastOp>(
          loc, llvmTypes.llvmInt32Type, toStore);
      Value storeAddress = rewriter.create<LLVM::GEPOp>(
          loc,
          LLVM::LLVMPointerType::get(llvmTypes.llvmInt32Type,
                                     /*NVVM private memory space*/ 0),
          storeAddressCasted,
          ArrayRef<Value>{rewriter.create<LLVM::ConstantOp>(
              loc, llvmTypes.llvmInt32Type, rewriter.getUI32IntegerAttr(i))});
      rewriter.create<LLVM::StoreOp>(loc, toStoreI32, storeAddress);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  CommonLLVMTypes llvmTypes;
};
} // namespace mlir
