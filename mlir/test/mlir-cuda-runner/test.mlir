// RUN: mlir-cuda-runner %s --shared-libs=%cuda_wrapper_library_dir/libcuda-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

func @main() {
  %c8 = llvm.mlir.constant(8) : !llvm.i64
  %res = llvm.alloca %c8 x !llvm.float : (!llvm.i64) -> !llvm<"float*">

  %data = alloc() : memref<2x6xi32>
  %sum = alloc() : memref<2xi32>
  %ref  = alloc() : memref<2xi32>
  %cst0 = constant 0 : i32
  %cst1 = constant 1 : i32
  %cst2 = constant 2 : i32
  %cst4 = constant 4 : i32
  %cst8 = constant 8 : i32
  %cst16 = constant 16 : i32

  %cst3 = constant 3 : i32
  %cst6 = constant 6 : i32
  %cst7 = constant 7 : i32
  %cst10 = constant 10 : i32
  %cst11 = constant 11 : i32

  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c5 = constant 5 : index
  %c6 = constant 6 : index

  //%cast_res = memref_cast %res : memref<2xi32> to memref<*xi32>
  //call @mgpuMemHostRegisterInt32(%cast_sum) : (memref<*xi32>) -> ()
  %cast_data = memref_cast %data : memref<2x6xi32> to memref<*xi32>
  call @mgpuMemHostRegisterInt32(%cast_data) : (memref<*xi32>) -> ()
  %cast_sum = memref_cast %sum : memref<2xi32> to memref<*xi32>
  call @mgpuMemHostRegisterInt32(%cast_sum) : (memref<*xi32>) -> ()
  %cast_ref = memref_cast %ref : memref<2xi32> to memref<*xi32>
  call @mgpuMemHostRegisterInt32(%cast_ref) : (memref<*xi32>) -> ()

  store %cst0, %data[%c0, %c0] : memref<2x6xi32>
  store %cst1, %data[%c0, %c1] : memref<2x6xi32>
  store %cst2, %data[%c0, %c2] : memref<2x6xi32>
  store %cst4, %data[%c0, %c3] : memref<2x6xi32>
  store %cst8, %data[%c0, %c4] : memref<2x6xi32>
  store %cst16, %data[%c0, %c5] : memref<2x6xi32>

  store %cst2, %data[%c1, %c0] : memref<2x6xi32>
  store %cst3, %data[%c1, %c1] : memref<2x6xi32>
  store %cst6, %data[%c1, %c2] : memref<2x6xi32>
  store %cst7, %data[%c1, %c3] : memref<2x6xi32>
  store %cst10, %data[%c1, %c4] : memref<2x6xi32>
  store %cst11, %data[%c1, %c5] : memref<2x6xi32>

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c2, %grid_y = %c1, %grid_z = %c1)
  threads(%tx, %ty, %tz) in (%block_x = %c6, %block_y = %c1, %block_z = %c1) {

    //%a0 = llvm.mlir.constant(sparse<[[0]], [1.0e+00]> : vector<2xf16>) : !llvm<"<2 x half>">
    //%a1 = llvm.mlir.constant(sparse<[[0]], [1.0e+00]> : vector<2xf16>) : !llvm<"<2 x half>">
    //%b0 = llvm.mlir.constant(sparse<[[0]], [1.0e+00]> : vector<2xf16>) : !llvm<"<2 x half>">
    //%b1 = llvm.mlir.constant(sparse<[[0]], [1.0e+00]> : vector<2xf16>) : !llvm<"<2 x half>">
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

    %c = llvm.extractvalue %0[0] : !llvm<"{ float, float, float, float, float, float, float, float }">
    //llvm.call @print_f32(%c) : (!llvm.float) -> ()

    //%c = llvm.mlir.constant(0) : !llvm.i32
    //%1 = llvm.getelementptr %0[%c] : (!llvm<"float* ">, !llvm.i32) -> !llvm.float

    %val = load %data[%bx, %tx] : memref<2x6xi32>
    %reduced = "gpu.all_reduce"(%val) ({}) { op = "max" } : (i32) -> (i32)
    store %reduced, %sum[%bx] : memref<2xi32>
    //store %c, llvm.extractvalue %0[0] : !llvm<"{ float, float, float, float, float, float, float, float }">
    gpu.terminator
  }


  call @print_memref_i32(%cast_sum) : (memref<*xi32>) -> ()
  // CHECK: [16, 11]
  call @print_memref_i32(%cast_ref) : (memref<*xi32>) -> ()

  return
}

func @mgpuMemHostRegisterInt32(%ptr : memref<*xi32>)
func @print_memref_i32(memref<*xi32>)

