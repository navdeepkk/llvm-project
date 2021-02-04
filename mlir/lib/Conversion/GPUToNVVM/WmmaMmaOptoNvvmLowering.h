//===--- WmmaMmaOptoNVVMLowering.h - MmaOp to NVVM Op lowering -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains patterns to lower the GPU subgroup MMA_computeop to the
// NVVM Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_CONVERSION_GPUTONVVM_WMMASTOREOPTONVVMLOWERING_H
#define MLIR_LIB_CONVERSION_GPUTONVVM_WMMASTOREOPTONVVMLOWERING_H

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
  MLIRContext *context = &this->getTypeConverter()->getContext();

  explicit WmmaMmaOptoNVVMLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<gpu::SubgroupMmaComputeOp>(typeConverter),
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
  matchAndRewrite(gpu::SubgroupMmaComputeOp subgroupMmaComputeOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *op = subgroupMmaComputeOp.getOperation();
    if (failed(areAllLLVMTypes(op, operands, rewriter)))
      return failure();

    Location loc = op->getLoc();

    SmallVector<mlir::VectorType, 4> opTypes;
    SmallVector<Type, 4> elemTypes;
    SmallVector<Type, 4> llvmElemTypes;
    SmallVector<Value, 4> opIndices;

    // The wmma.mma intrinsic in llvm requires the operands as individual
    // values. So individual elements from the memrefs need to be extracted and
    // then passed on to the intrinsic call. Emit llvm ops to extract individual
    // values form lowered memrefs.
    SmallVector<Value, 24> unpackedOps;

    auto unpackOp = [&](CommonLLVMTypes::OperandMap op, Value operand) {
      for (unsigned i = 0, e = llvmTypes.numHalfsInOpFrags[op]; i < e; ++i) {
        Value toUse = rewriter.create<LLVM::ExtractValueOp>(
            loc, llvmTypes.f16x2Ty, operand, rewriter.getI64ArrayAttr(i));
        unpackedOps.push_back(toUse);
      }
    };

    unpackOp(llvmTypes.A, operands[0]);
    unpackOp(llvmTypes.B, operands[1]);
    unpackOp(llvmTypes.C, operands[2]);

    // Operand holder for wmma.mma.op.
    ValueRange wmmaMmaOpOperands(unpackedOps);

    // Create nvvm.wmma.mma op.
    NVVM::WMMAMmaOp wmmaMmaOp = rewriter.create<NVVM::WMMAMmaOp>(
        loc, llvmTypes.fragArrayCDTy, wmmaMmaOpOperands);

    rewriter.replaceOp(op, wmmaMmaOp.getResult());
    return success();
  }

private:
  /// Contains definitions of all the LLVM types which are used for lowering
  /// this GPU subgroupMmaComputeOp.
  CommonLLVMTypes llvmTypes;
};
} // namespace mlir

#endif
