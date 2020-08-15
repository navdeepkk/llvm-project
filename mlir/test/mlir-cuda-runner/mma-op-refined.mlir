// RUN: mlir-cuda-runner %s --shared-libs=%cuda_wrapper_library_dir/libcuda-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

func @main() {
  %c4 = llvm.mlir.constant(4) : !llvm.i32
  %c8 = llvm.mlir.constant(8) : !llvm.i32
  %c84 = llvm.mul %c8, %c4 : !llvm.i32
  %c48 = llvm.mul %c4, %c8 : !llvm.i32
  %c88 = llvm.mul %c8, %c8 : !llvm.i32
  %A = llvm.alloca %c84 x !llvm.half : (!llvm.i32) -> !llvm<"half*">
  %B = llvm.alloca %c48 x !llvm.half : (!llvm.i32) -> !llvm<"half*">
  %C = llvm.alloca %c88 x !llvm.half : (!llvm.i32) -> !llvm<"half*">
  %D = llvm.alloca %c88 x !llvm.half : (!llvm.i32) -> !llvm<"half*">
  
  %c1 = constant 1 : index
  %c32 = constant 32 : index
  %c16 = constant 16 : i32
  %c4i = constant 4 : i32

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
  threads(%tx, %ty, %tz) in (%block_x = %c32, %block_y = %c1, %block_z = %c1) {

    %lane = index_cast %tx : index to i32

    %lnlt16 = std.cmpi "ult", %lane, %c16 : i32 
    
    scf.if %lnlt16 {
      %outer = std.remi_unsigned %lane, %c4i : i32
    }else{
      %temp = std.remi_unsigned %lane, %c4i : i32
      %outer = std.addi %temp, %c4i : i32
    }
    // Part to go extract the input fragments for different threads goes here. 
    //---------------------------------------------------------------------//
    %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7 {alayout="row", blayout="col"} : (!llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float) -> !llvm<"{ float, float, float, float, float, float, float, float }">

    %res0 = llvm.extractvalue %0[0] : !llvm<"{ float, float, float, float, float, float, float, float }"> 
    gpu.terminator
  }

  return
}

