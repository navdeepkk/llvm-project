// RUN: mlir-cuda-runner %s --shared-libs=%cuda_wrapper_library_dir/libcuda-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

llvm.func @main() {
  %c8 = llvm.mlir.constant(8) : !llvm.i64
  %res = llvm.alloca %c8 x !llvm.float : (!llvm.i64) -> !llvm<"float*">
  
  %c1 = constant 1 : index
  %c32 = constant 32 : index

  //%cast_res = memref_cast %res : memref<2xi32> to memref<*xi32>
  //call @mgpuMemHostRegisterInt32(%cast_sum) : (memref<*xi32>) -> ()

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
  threads(%tx, %ty, %tz) in (%block_x = %c32, %block_y = %c1, %block_z = %c1) {

    %a0 = llvm.mlir.constant(dense<1.0e+00> : vector<2xf16>) : !llvm<"<2 x half>">
    %a1 = llvm.mlir.constant(dense<1.0e+00> : vector<2xf16>) : !llvm<"<2 x half>">
    %b0 = llvm.mlir.constant(dense<1.0e+00> : vector<2xf16>) : !llvm<"<2 x half>">
    %b1 = llvm.mlir.constant(dense<1.0e+00> : vector<2xf16>) : !llvm<"<2 x half>">
    %d0 = llvm.mlir.constant(0.000000e+00 : f32) : !llvm.float
    %d1 = llvm.mlir.constant(0.000000e+00 : f32) : !llvm.float
    %d2 = llvm.mlir.constant(0.000000e+00 : f32) : !llvm.float
    %d3 = llvm.mlir.constant(0.000000e+00 : f32) : !llvm.float
    %d4 = llvm.mlir.constant(0.000000e+00 : f32) : !llvm.float
    %d5 = llvm.mlir.constant(0.000000e+00 : f32) : !llvm.float
    %d6 = llvm.mlir.constant(0.000000e+00 : f32) : !llvm.float
    %d7 = llvm.mlir.constant(0.000000e+00 : f32) : !llvm.float

    %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7 {alayout="row", blayout="col"} : (!llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float) -> !llvm<"{ float, float, float, float, float, float, float, float }">

    //llvm.call @print_f32(%c) : (!llvm.float) -> ()

    //%c = llvm.mlir.constant(0) : !llvm.i32
    //%1 = llvm.getelementptr %0[%c] : (!llvm<"float* ">, !llvm.i32) -> !llvm.float
    %res0 = llvm.extractvalue %0[0] : !llvm<"{ float, float, float, float, float, float, float, float }"> 
    gpu.terminator
  }

  return
}
