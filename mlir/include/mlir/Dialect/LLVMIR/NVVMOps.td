//===-- NVVMOps.td - NVVM IR dialect op definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the NVVM IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifndef NVVMIR_OPS
#define NVVMIR_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/Dialect/GPU/GPUBase.td"
//===----------------------------------------------------------------------===//
// NVVM dialect definitions
//===----------------------------------------------------------------------===//

def NVVM_Dialect : Dialect {
  let name = "nvvm";
  let cppNamespace = "::mlir::NVVM";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// NVVM op definitions
//===----------------------------------------------------------------------===//

class NVVM_Op<string mnemonic, list<OpTrait> traits = []> :
  LLVM_OpBase<NVVM_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// NVVM intrinsic operations
//===----------------------------------------------------------------------===//

class NVVM_IntrOp<string mnem, list<int> overloadedResults,
                  list<int> overloadedOperands, list<OpTrait> traits,
                  int numResults>
  : LLVM_IntrOpBase<NVVM_Dialect, mnem, "nvvm_" # !subst(".", "_", mnem),
                    overloadedResults, overloadedOperands, traits, numResults>;


//===----------------------------------------------------------------------===//
// NVVM special register op definitions
//===----------------------------------------------------------------------===//

class NVVM_SpecialRegisterOp<string mnemonic,
    list<OpTrait> traits = []> :
  NVVM_IntrOp<mnemonic, [], [], !listconcat(traits, [NoSideEffect]), 1>,
  Arguments<(ins)> {
  let assemblyFormat = "attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Lane index and range
def NVVM_LaneIdOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.laneid">;
def NVVM_WarpSizeOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.warpsize">;

//===----------------------------------------------------------------------===//
// Thread index and range
def NVVM_ThreadIdXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.tid.x">;
def NVVM_ThreadIdYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.tid.y">;
def NVVM_ThreadIdZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.tid.z">;
def NVVM_BlockDimXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ntid.x">;
def NVVM_BlockDimYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ntid.y">;
def NVVM_BlockDimZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ntid.z">;

//===----------------------------------------------------------------------===//
// Block index and range
def NVVM_BlockIdXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ctaid.x">;
def NVVM_BlockIdYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ctaid.y">;
def NVVM_BlockIdZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ctaid.z">;
def NVVM_GridDimXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nctaid.x">;
def NVVM_GridDimYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nctaid.y">;
def NVVM_GridDimZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nctaid.z">;

//===----------------------------------------------------------------------===//
// NVVM synchronization op definitions
//===----------------------------------------------------------------------===//

def NVVM_Barrier0Op : NVVM_Op<"barrier0"> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_barrier0);
  }];
  let assemblyFormat = "attr-dict";
}

def NVVM_ShflBflyOp :
  NVVM_Op<"shfl.sync.bfly">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_Type:$dst,
                 LLVM_Type:$val,
                 LLVM_Type:$offset,
                 LLVM_Type:$mask_and_clamp,
                 OptionalAttr<UnitAttr>:$return_value_and_is_valid)> {
  string llvmBuilder = [{
      auto intId = getShflBflyIntrinsicId(
          $_resultType, static_cast<bool>($return_value_and_is_valid));
      $res = createIntrinsicCall(builder,
          intId, {$dst, $val, $offset, $mask_and_clamp});
  }];
  let parser = [{ return parseNVVMShflSyncBflyOp(parser, result); }];
  let printer = [{ printNVVMIntrinsicOp(p, this->getOperation()); }];
  let verifier = [{
    if (!(*this)->getAttrOfType<UnitAttr>("return_value_and_is_valid"))
      return success();
    auto type = getType().dyn_cast<LLVM::LLVMStructType>();
    auto elementType = (type && type.getBody().size() == 2)
                     ? type.getBody()[1].dyn_cast<LLVM::LLVMIntegerType>()
                     : nullptr;
    if (!elementType || elementType.getBitWidth() != 1)
      return emitError("expected return type to be a two-element struct with "
                       "i1 as the second element");
    return success();
  }];
}

def NVVM_VoteBallotOp :
  NVVM_Op<"vote.ballot.sync">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_Type:$mask, LLVM_Type:$pred)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder,
            llvm::Intrinsic::nvvm_vote_ballot_sync, {$mask, $pred});
  }];
  let parser = [{ return parseNVVMVoteBallotOp(parser, result); }];
  let printer = [{ printNVVMIntrinsicOp(p, this->getOperation()); }];
}

def NVVM_MmaOp :
  NVVM_Op<"mma.sync">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins Variadic<LLVM_Type>:$args)> {
  string llvmBuilder = [{
    $res = createIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_mma_m8n8k4_row_col_f32_f32, $args);
  }];
  let assemblyFormat = "$args attr-dict `:` functional-type($args, $res)";
  let verifier = [{ return ::verify(*this); }];
}

def NVVM_WMMALoadOp :
  NVVM_Op<"wmma.load">,
  Results<(outs LLVM_AnyStruct:$res)>,
  Arguments<(ins Variadic<LLVM_Type>:$args,
            UI16Attr:$wm,
            UI16Attr:$wn,
            UI16Attr:$wk,
            I64Attr:$ldm,
            SubgroupMmaOperand:$operand)> {
  string llvmBuilder = [{
    if($operand.compare("AOp") == 0)
      $res = createNvvmIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_load_a_f16_row_stride, $args);

    else if($operand.compare("BOp") == 0)
      $res = createNvvmIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_load_b_f16_row_stride, $args);

    else if($operand.compare("COp") == 0)
      $res = createNvvmIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_load_c_f16_row_stride, $args);
  }];
 
  let summary = "Warp synchronous matrix load";

  let description = [{ 
    The `nvvm.wmma.load` operation loads a matrix collectively using all the threads
    in a warp.

    The operation takes two arguments, the address from where the matrix elements are
    to be loaded from and a stride. The stride argument represents the leading dimension
    of the source matrix. The address and the stride are required to be the same across
    all threads in the warp. The Shape and type of the matrix being loaded are specified
    using the attribiutes `wm`, `wn`, `wk` and `operand`. Each thread in a warp holds a
    certain number of elements. The Op returns a LLVM struct which holds the elements of
    the matrix held by this thread.

    This op is meant to be used along with `nvvm.wmma.store` and `nvvm.wmma.mma`.

    Example:
    
    ```mlir
    %2 = nvvm.wmma.load %0, %1 {ldm = 32 : i64, operand = "AOp", wk = 16 :
    ui16, wm = 16 : ui16, wn = 16 : ui16} : !llvm.ptr<i32, 3>, !llvm.i32 ->
    !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>,
    vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    ```
  }];
  
  let assemblyFormat = "$args attr-dict `:` type($args) `->` type($res)";

  let verifier = [{ return ::verify(*this); }];
}

def NVVM_WMMAStoreOp :
  NVVM_Op<"wmma.store">,
  Arguments<(ins Variadic<LLVM_Type>:$args,
            UI16Attr:$wm,
            UI16Attr:$wn,
            UI16Attr:$wk,
            I64Attr:$ldm)> {
  string llvmBuilder = [{
        createNvvmIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_store_d_f16_row_stride, $args);
  }];

  let summary = "Warp synchronous matrix store";
  
  let description = [{ 
    The `nvvm.wmma.store` operation stores a matrix collectively using all the threads
    in a warp.

    The operation takes as arguments the address to where the matrix elements are
    to be stored, a stride and the elements to store, held by the current thread.
    The stride argument represents the leading dimension of the destination matrix.
    The address and the stride are required to be the same across all threads in the
    warp. The Shape of the matrix being stored are specified using the attribiutes
    `wm`, `wn`, `wk`.

    This op is meant to be used along with `nvvm.wmma.load` and `nvvm.wmma.mma`.

    Example:
    
    ```mlir
    nvvm.wmma.store %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10{ldm = 32 : i64,
    , wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : !llvm.ptr<i32, 3>
    !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>,
    vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>, !llvm.i32
    ```
  }];
  
  let assemblyFormat = "$args attr-dict `:` type($args)";

  let verifier = [{ return ::verify(*this); }];
}

def NVVM_WMMAMmaOp :
  NVVM_Op<"wmma.mma">,
  Results<(outs LLVM_AnyStruct:$res)>,
  Arguments<(ins Variadic<LLVM_Type>:$args,
            UI16Attr:$wm,
            UI16Attr:$wn,
            UI16Attr:$wk)> {
  string llvmBuilder = [{
    $res = createNvvmIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_row_row_f16_f16, $args);
  }];

  let summary = "Warp synchronous matrix-multiply accumulate using tensor cores.";

  let description = [{ 
    The `nvvm.wmma.stores` operation performs a matrix-multiply accumulate(mma)
    operation using all the threads in a warp.

    The operation performed is represented as `D = A * B + C`. The operation takes
    as arguments the elements of the matrices `A`, `B`, `C` and `D`, held by the
    current thread. The Shape of the overall operation is represented using the
    attribiutes `wm`, `wn`, `wk`. The op returns a LLVM struct which holds a part
    of the result held by the current thread. 

    This op is meant to be used along with `nvvm.wmma.load` and `nvvm.wmma.store`.

    Example:
    
    ```mlir
    %20 = nvvm.wmma.mma %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12,
    %13, %14, %15, %16, %17, %18, %19 {wk = 16 : ui16, wm = 16 : ui16, wn = 16 :
    ui16} : !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>,
    !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>,
    !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>,
    !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>,
    !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>,
    !llvm.vec<2 x half> -> !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>,
    vec<2 x half>)>
    ```
  }];
  
  let assemblyFormat = "$args attr-dict `:` type($args) `->` type($res)";

  let verifier = [{ return ::verify(*this); }];
}

#endif // NVVMIR_OPS
