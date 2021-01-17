//===--- WmmaMmaOptoNVVMLowering.h - GPU MMA mmaOp to NVVM Op lowering ----===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the patterns to lower the GPU subgroup MMA ops to the NVVM
// Dialect.
//
//===----------------------------------------------------------------------===//

#include "CommonTypes.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

namespace mlir {
/// This class implemtents the conversion of GPU MMA computeOp to wmma.mma op
/// in the NVVM dialect. The conversion not only emits the NVVM op but also
/// emits code that is necessary to unpack the data from source memrefs to give
/// them to the NVVM OP and then again pack the results to store them into the
/// destination memref.
struct WmmaMmaOptoNVVMLowering
    : public ConvertOpToLLVMPattern<gpu::SubgroupMmaComputeOp> {
public:
  enum operandMap { A, B, C, D };
  MLIRContext *context = &this->getTypeConverter()->getContext();

  explicit WmmaMmaOptoNVVMLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<gpu::SubgroupMmaComputeOp>(typeConverter),
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
  matchAndRewrite(gpu::SubgroupMmaComputeOp subgroupMmaComputeOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *op = subgroupMmaComputeOp.getOperation();
    if (failed(areAllLLVMTypes(op, operands, rewriter)))
      return failure();

    Location loc = op->getLoc();

    SmallVector<MemRefType, 4> opTypes;
    SmallVector<Type, 4> elemTypes;
    SmallVector<ArrayRef<int64_t>, 4> opShapes;

    auto populateOpInfo = [&opTypes, &elemTypes, &opShapes,
                           &subgroupMmaComputeOp]() {
      opTypes.push_back(
          subgroupMmaComputeOp.opA().getType().cast<MemRefType>());
      opTypes.push_back(
          subgroupMmaComputeOp.opB().getType().cast<MemRefType>());
      opTypes.push_back(
          subgroupMmaComputeOp.opC().getType().cast<MemRefType>());
      opTypes.push_back(
          subgroupMmaComputeOp.opD().getType().cast<MemRefType>());

      for (MemRefType opType : opTypes) {
        elemTypes.push_back(opType.getElementType());
        opShapes.push_back(opType.getShape());
      }
    };
    // Gather type, shape info fo the memrefs.
    populateOpInfo();

    // Promote operands of this op. This emits !llvm.extractelement for each of
    // the operand memrefs and makes it easy to use these values in subsequent
    // instruction.
    SmallVector<SmallVector<Value, 4>, 4> promotedOps;
    promotedOps.resize(4);

    auto promoteOps = [&](operandMap operand) {
      promotedOps[operand] = this->getTypeConverter()->promoteOperands(
          loc, op->getOperand(operand), operands[operand],
          rewriter);
    };

    promoteOps(A);
    promoteOps(B);
    promoteOps(C);
    promoteOps(D);

    // The wmma.mma intrinsic in llvm requires the operands as individual
    // values. So individual elements from the memrefs need to be extracted and
    // then passed on to the intrinsic call. Emit llvm ops to extract individual
    // values form lowered memrefs.
    SmallVector<Value, 24> unpackedOps;

    auto unpackOp = [&](operandMap op) {
      for (unsigned i = 0, e = opShapes[op][0]; i < e; ++i) {
        Value loadAddress = rewriter.create<LLVM::GEPOp>(
            loc,
            LLVM::LLVMPointerType::get(llvmTypes.llvmF16x2Ty,
                                       /*NVVM private memory space*/ 0),
            promotedOps[op][1],
            ArrayRef<Value>{rewriter.create<LLVM::ConstantOp>(
                loc, llvmTypes.llvmInt32Type, rewriter.getUI32IntegerAttr(i))});

        unpackedOps.push_back(rewriter.create<LLVM::LoadOp>(loc, loadAddress));
      }
    };

    unpackOp(A);
    unpackOp(B);
    unpackOp(C);

    auto wm = subgroupMmaComputeOp.wmAttr();
    auto wn = subgroupMmaComputeOp.wnAttr();
    auto wk = subgroupMmaComputeOp.wkAttr();

    // Operand holder for wmma.mma.op.
    ValueRange wmmaMmaOpOperands(unpackedOps);

    // Create nvvm.wmma.mma op.
    NVVM::WMMAMmaOp wmmaMmaOp = rewriter.create<NVVM::WMMAMmaOp>(
        loc, llvmTypes.fragArrayCDTy, wmmaMmaOpOperands,
        wm.cast<mlir::IntegerAttr>(), wn.cast<mlir::IntegerAttr>(),
        wk.cast<mlir::IntegerAttr>());

    // Bitcast the base address pointer of the destination memref, So that
    // values can be stored in chunks of 32-bits, as they were returned by the
    // wmmaLoadOP.
    Value storeAddressCasted = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(llvmTypes.llvmInt32Type,
                                   /*NVVM private memory space*/ 0),
        promotedOps[D][1]);

    // Store the results in memref D.
    for (unsigned i = 0, e = opShapes[D][0]; i < e; ++i) {
      Value toStore = rewriter.create<LLVM::ExtractValueOp>(
          loc, llvmTypes.llvmF16x2Ty, wmmaMmaOp, rewriter.getIndexArrayAttr(i));
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
