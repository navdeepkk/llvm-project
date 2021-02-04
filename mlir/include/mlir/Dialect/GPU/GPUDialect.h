//===- GPUDialect.h - MLIR Dialect for GPU Kernels --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the GPU kernel-related operations and puts them in the
// corresponding dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_GPUDIALECT_H
#define MLIR_DIALECT_GPU_GPUDIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class FuncOp;

namespace gpu {

/// Utility class for the GPU dialect to represent triples of `Value`s
/// accessible through `.x`, `.y`, and `.z` similarly to CUDA notation.
struct KernelDim3 {
  Value x;
  Value y;
  Value z;
};

class AsyncTokenType
    : public Type::TypeBase<AsyncTokenType, Type, TypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;
};

/// MMAFragmentType storage and uniquing.
struct MMAFragmentStorageType : public TypeStorage {
  MMAFragmentStorageType(int64_t size, Type elementType)
      : size(size), elementType(elementType) {}

  /// The hash key for uniquing.
  using KeyTy = std::pair<int64_t, Type>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(size, elementType);
  }

  /// Construction.
  static MMAFragmentStorageType *construct(TypeStorageAllocator &allocator,
                                           const KeyTy &key) {
    return new (allocator.allocate<MMAFragmentStorageType>())
        MMAFragmentStorageType(key.first, key.second);
  }

  /// Number of elements held in the fragment.
  int64_t size;

  /// Element type of elements held in the fragment.
  Type elementType;
};

/// MMAFragment represents a fragment or collection of elements held by a thread
/// for matrix-matrix multiply accumulate operations. MMAFragments are taken as
/// direct operands by these operations and are also produced as results. There
/// fragments are meant to reside in the registers. A limited number of
/// pointwise operations can be performed on these fragments, i.e., operations
/// which operate uniformly on all the elements in the fragment and do not
/// change the order of matrix elements in the fragments. The above conditions
/// exist because the layout of matrix elemnets inside the fragment is opaque
/// i.e., the elements may be present in the fragment in any order.
class MMAFragmentType
    : public Type::TypeBase<MMAFragmentType, Type, MMAFragmentStorageType> {
public:
  using Base::Base;

  /// Get MMAFragmentType and verify construction Invariants.
  static MMAFragmentType get(int64_t shape, Type elementType);

  /// Get MMAFragmentType at a particular location and verify construction
  /// Invariants.
  static MMAFragmentType
  getChecked(function_ref<InFlightDiagnostic()> emitError, int64_t shape,
             Type elementType);

  /// Check if a type is valid a MMAFragmentType elementType.
  static bool isValidElementType(Type elementType);

  /// Verify that shape and elementType are actually allowed for the
  /// MMAFragmentType.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError, int64_t shape,
                                                    Type elementType);

  /// Get size of MMAFragment in number of elements.
  int64_t getSize() const;

  /// Get elementType of a single element in MMAFragment.
  Type getElementType() const;
};

// Adds a `gpu.async.token` to the front of the argument list.
void addAsyncDependency(Operation *op, Value token);

} // end namespace gpu
} // end namespace mlir

#include "mlir/Dialect/GPU/GPUOpsDialect.h.inc"

#include "mlir/Dialect/GPU/GPUOpInterfaces.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/GPU/GPUOps.h.inc"

#endif // MLIR_DIALECT_GPU_GPUDIALECT_H
