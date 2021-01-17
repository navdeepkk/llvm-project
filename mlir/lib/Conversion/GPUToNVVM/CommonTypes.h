//===----- CommonTypes.h - Contains LLVM Types common to all Lowerings. ---===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the common LLVM Types that are used by the lowerings of
// GPU MMA Ops to NVVM ops.
//
//===----------------------------------------------------------------------===//
#ifndef COMMON_TYPES_INCLUDED
#define COMMON_TYPES_INCLUDED

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/IR/DerivedTypes.h"

namespace mlir {

struct CommonLLVMTypes {
public:
  CommonLLVMTypes(MLIRContext *context) {
    llvmInt8Type = LLVM::LLVMType::getInt8Ty(context);
    llvmInt64Type = LLVM::LLVMType::getInt64Ty(context);
    llvmInt32Type = LLVM::LLVMType::getInt32Ty(context);
    llvmInt32PtrTy = LLVM::LLVMPointerType::get(llvmInt32Type);
    llvmF16Ty = LLVM::LLVMType::getHalfTy(context);
    llvmF16PtrTy = LLVM::LLVMPointerType::get(llvmF16Ty);
    llvmF16x2Ty = LLVM::LLVMType::getVectorTy(llvmF16Ty, 2);
    fragArrayABTy = LLVM::LLVMType::getStructTy(
        context, SmallVector<LLVM::LLVMType>(8, llvmF16x2Ty));
    fragArrayABPtrTy = LLVM::LLVMPointerType::get(fragArrayABTy);
    fragArrayCDTy = LLVM::LLVMType::getStructTy(
        context, SmallVector<LLVM::LLVMType>(4, llvmF16x2Ty));
    fragArrayCDPtrTy = LLVM::LLVMPointerType::get(fragArrayCDTy);
  };

  LLVM::LLVMType llvmInt8Type;
  LLVM::LLVMType llvmInt32Type;
  LLVM::LLVMType llvmInt64Type;
  LLVM::LLVMType llvmInt32PtrTy;
  LLVM::LLVMType llvmF16Ty;
  LLVM::LLVMType llvmF16PtrTy;
  LLVM::LLVMType llvmF16x2Ty;
  LLVM::LLVMType fragArrayABTy;
  LLVM::LLVMType fragArrayABPtrTy;
  LLVM::LLVMType fragArrayCDTy;
  LLVM::LLVMType fragArrayCDPtrTy;
};

} // namespace mlir
#endif
