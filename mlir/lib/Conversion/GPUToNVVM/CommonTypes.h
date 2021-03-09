//===----- CommonTypes.h - Contains LLVM Types common to all Lowerings. ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the common LLVM types that are used by the lowerings of
// GPU MMA Ops to NVVM ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_CONVERSION_GPUTONVVM_COMMONTYPES_H
#define MLIR_LIB_CONVERSION_GPUTONVVM_COMMONTYPES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/IR/DerivedTypes.h"

namespace mlir {

/// Contains all the common LLVM types which are used across the lowerings of
/// GPU subgroup ops to NVVM dialect.
struct CommonLLVMTypes {
public:
  CommonLLVMTypes(MLIRContext *context) {
    numHalfsInOpFrags.resize(4);
    numHalfsInOpFrags[A] = 8;
    numHalfsInOpFrags[B] = 8;
    numHalfsInOpFrags[C] = 4;
    numHalfsInOpFrags[D] = 4;
    int8Type = IntegerType::get(context, 8);
    int64Type = IntegerType::get(context, 64);
    int32Type = IntegerType::get(context, 32);
    int32PtrTy = LLVM::LLVMPointerType::get(int32Type);
    f16Ty = FloatType::getF16(context);
    f32Ty = FloatType::getF32(context);
    f16PtrTy = LLVM::LLVMPointerType::get(f16Ty);
    f16x8Ty = VectorType::get(8, f16Ty);
    f16x16Ty = VectorType::get(16, f16Ty);
    f16x2Ty = VectorType::get(2, f16Ty);
    fragArrayABTy = LLVM::LLVMStructType::getLiteral(
        context, SmallVector<Type>(8, f16x2Ty));
    fragArrayABPtrTy = LLVM::LLVMPointerType::get(fragArrayABTy);
    fragArrayCDTy = LLVM::LLVMStructType::getLiteral(
        context, SmallVector<Type>(4, f16x2Ty));
    fragArrayCDF32Ty =
        LLVM::LLVMStructType::getLiteral(context, SmallVector<Type>(8, f32Ty));
  };

  Type int8Type;
  Type int32Type;
  Type int64Type;
  Type int32PtrTy;
  Type f16Ty;
  Type f32Ty;
  Type f16PtrTy;
  Type f16x2Ty;
  Type f16x8Ty;
  Type f16x16Ty;
  /// Type for the fragment of A and B operands that a single thread holds for
  /// fp16 data type.
  Type fragArrayABTy;
  /// Type for a pointer to the fragment of `AB` operands that a single thread
  /// holds for fp16 data type.
  Type fragArrayABPtrTy;
  /// Type for the fragment of C and D operands that a single thread holds for
  /// fp16 data type.
  Type fragArrayCDTy;
  /// Type for the fragment of C and D operands that a single thread holds for
  /// fp32 data type.
  Type fragArrayCDF32Ty;
  SmallVector<unsigned, 4> numHalfsInOpFrags;
  enum OperandMap { A, B, C, D };
};

} // namespace mlir
#endif
