//===--- WmmaMmaOpToNVVMLowering.h - MmaOp to NVVM Op lowering -*- C++ -*-===//
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
struct WmmaMmaOpToNVVMLowering
    : public ConvertOpToLLVMPattern<gpu::SubgroupMmaComputeOp> {
public:
  MLIRContext *context = &this->getTypeConverter()->getContext();

  explicit WmmaMmaOpToNVVMLowering(LLVMTypeConverter &typeConverter)
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

    // The wmma.mma intrinsic in llvm requires the operands as individual
    // values. So individual elements from the memrefs need to be extracted and
    // then passed on to the intrinsic call. Emit llvm ops to extract individual
    // values form lowered memrefs.
    SmallVector<Value, 24> unpackedOps;

    auto unpackOp = [&](CommonLLVMTypes::OperandMap op, Value operand,
                        unsigned numElems, Type elemType) {
      for (unsigned i = 0; i < numElems; ++i) {
        Value toUse = rewriter.create<LLVM::ExtractValueOp>(
            loc, elemType, operand, rewriter.getI64ArrayAttr(i));
        unpackedOps.push_back(toUse);
      }
    };

    if (subgroupMmaComputeOp.opC()
            .getType()
            .cast<gpu::MMAFragmentType>()
            .getElementType() == llvmTypes.f16x2Ty) {
      unpackOp(llvmTypes.A, operands[0],
               llvmTypes.numHalfsInOpFrags[llvmTypes.A], llvmTypes.f16x2Ty);
      unpackOp(llvmTypes.B, operands[1],
               llvmTypes.numHalfsInOpFrags[llvmTypes.B], llvmTypes.f16x2Ty);
      unpackOp(llvmTypes.C, operands[2],
               llvmTypes.numHalfsInOpFrags[llvmTypes.C], llvmTypes.f16x2Ty);

      // Create nvvm.wmma.mma op.
      NVVM::WMMAMmaF16F16Op wmmaMmaOp = rewriter.create<NVVM::WMMAMmaF16F16Op>(
          loc, llvmTypes.fragArrayCDTy, unpackedOps);

      rewriter.replaceOp(op, wmmaMmaOp.getResult());
      return success();
    }

    if (subgroupMmaComputeOp.opC()
            .getType()
            .cast<gpu::MMAFragmentType>()
            .getElementType() == llvmTypes.f32Ty) {
      unpackOp(llvmTypes.A, operands[0],
               llvmTypes.numHalfsInOpFrags[llvmTypes.A], llvmTypes.f16x2Ty);
      unpackOp(llvmTypes.B, operands[1],
               llvmTypes.numHalfsInOpFrags[llvmTypes.B], llvmTypes.f16x2Ty);
      unpackOp(llvmTypes.C, operands[2], 8, llvmTypes.f32Ty);

      // Create nvvm.wmma.mma op.
      NVVM::WMMAMmaF32F32Op wmmaMmaOp = rewriter.create<NVVM::WMMAMmaF32F32Op>(
          loc, llvmTypes.fragArrayCDF32Ty, unpackedOps);

      rewriter.replaceOp(op, wmmaMmaOp.getResult());
      return success();
    }

    return failure();
  }

private:
  /// Definitions of all the LLVM types which are used for lowering
  /// this GPU subgroupMmaComputeOp.
  CommonLLVMTypes llvmTypes;
};
} // namespace mlir

#endif
