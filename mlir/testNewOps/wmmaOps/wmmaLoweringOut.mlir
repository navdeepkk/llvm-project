module attributes {gpu.container_module}  {
  func @main() {
    %0 = alloc() : memref<32x32xf16, 3>
    %1 = alloc() : memref<32x32xf16, 3>
    %c1 = constant 1 : index
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%0 : memref<32x32xf16, 3>, %1 : memref<32x32xf16, 3>)
    return
  }
  gpu.module @main_kernel {
    llvm.func @main_kernel(%arg0: !llvm.ptr<half, 3>, %arg1: !llvm.ptr<half, 3>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.ptr<half, 3>, %arg8: !llvm.ptr<half, 3>, %arg9: !llvm.i64, %arg10: !llvm.i64, %arg11: !llvm.i64, %arg12: !llvm.i64, %arg13: !llvm.i64) attributes {gpu.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %8 = llvm.mlir.undef : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %13 = llvm.insertvalue %arg12, %12[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.insertvalue %arg11, %13[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %15 = llvm.insertvalue %arg13, %14[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %16 = llvm.mlir.constant(8 : index) : !llvm.i64
      %17 = llvm.mlir.constant(1 : index) : !llvm.i64
      %18 = llvm.mlir.null : !llvm.ptr<vec<2 x half>>
      %19 = llvm.getelementptr %18[%16] : (!llvm.ptr<vec<2 x half>>, !llvm.i64) -> !llvm.ptr<vec<2 x half>>
      %20 = llvm.ptrtoint %19 : !llvm.ptr<vec<2 x half>> to !llvm.i64
      %21 = llvm.alloca %20 x !llvm.vec<2 x half> {alignment = 32 : i64} : (!llvm.i64) -> !llvm.ptr<vec<2 x half>>
      %22 = llvm.mlir.undef : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %23 = llvm.insertvalue %21, %22[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %24 = llvm.insertvalue %21, %23[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %25 = llvm.mlir.constant(0 : index) : !llvm.i64
      %26 = llvm.insertvalue %25, %24[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %27 = llvm.insertvalue %16, %26[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %28 = llvm.insertvalue %17, %27[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %29 = llvm.mlir.constant(8 : index) : !llvm.i64
      %30 = llvm.mlir.constant(1 : index) : !llvm.i64
      %31 = llvm.mlir.null : !llvm.ptr<vec<2 x half>>
      %32 = llvm.getelementptr %31[%29] : (!llvm.ptr<vec<2 x half>>, !llvm.i64) -> !llvm.ptr<vec<2 x half>>
      %33 = llvm.ptrtoint %32 : !llvm.ptr<vec<2 x half>> to !llvm.i64
      %34 = llvm.alloca %33 x !llvm.vec<2 x half> {alignment = 32 : i64} : (!llvm.i64) -> !llvm.ptr<vec<2 x half>>
      %35 = llvm.mlir.undef : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %36 = llvm.insertvalue %34, %35[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %37 = llvm.insertvalue %34, %36[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %38 = llvm.mlir.constant(0 : index) : !llvm.i64
      %39 = llvm.insertvalue %38, %37[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %40 = llvm.insertvalue %29, %39[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %41 = llvm.insertvalue %30, %40[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %42 = llvm.mlir.constant(4 : index) : !llvm.i64
      %43 = llvm.mlir.constant(1 : index) : !llvm.i64
      %44 = llvm.mlir.null : !llvm.ptr<vec<2 x half>>
      %45 = llvm.getelementptr %44[%42] : (!llvm.ptr<vec<2 x half>>, !llvm.i64) -> !llvm.ptr<vec<2 x half>>
      %46 = llvm.ptrtoint %45 : !llvm.ptr<vec<2 x half>> to !llvm.i64
      %47 = llvm.alloca %46 x !llvm.vec<2 x half> {alignment = 32 : i64} : (!llvm.i64) -> !llvm.ptr<vec<2 x half>>
      %48 = llvm.mlir.undef : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %49 = llvm.insertvalue %47, %48[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %50 = llvm.insertvalue %47, %49[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %51 = llvm.mlir.constant(0 : index) : !llvm.i64
      %52 = llvm.insertvalue %51, %50[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %53 = llvm.insertvalue %42, %52[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %54 = llvm.insertvalue %43, %53[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %55 = llvm.mlir.constant(4 : index) : !llvm.i64
      %56 = llvm.mlir.constant(1 : index) : !llvm.i64
      %57 = llvm.mlir.null : !llvm.ptr<vec<2 x half>>
      %58 = llvm.getelementptr %57[%55] : (!llvm.ptr<vec<2 x half>>, !llvm.i64) -> !llvm.ptr<vec<2 x half>>
      %59 = llvm.ptrtoint %58 : !llvm.ptr<vec<2 x half>> to !llvm.i64
      %60 = llvm.alloca %59 x !llvm.vec<2 x half> {alignment = 32 : i64} : (!llvm.i64) -> !llvm.ptr<vec<2 x half>>
      %61 = llvm.mlir.undef : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %62 = llvm.insertvalue %60, %61[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %63 = llvm.insertvalue %60, %62[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %64 = llvm.mlir.constant(0 : index) : !llvm.i64
      %65 = llvm.insertvalue %64, %63[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %66 = llvm.insertvalue %55, %65[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %67 = llvm.insertvalue %56, %66[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %68 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %69 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %70 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %71 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %72 = llvm.extractvalue %7[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %73 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %74 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %75 = llvm.extractvalue %28[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %76 = llvm.extractvalue %28[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %77 = llvm.extractvalue %28[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %78 = llvm.extractvalue %28[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %79 = llvm.extractvalue %28[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %80 = llvm.mlir.constant(16 : index) : !llvm.i32
      %81 = llvm.mlir.constant(16 : index) : !llvm.i32
      %82 = llvm.mlir.constant(16 : ui16) : !llvm.i32
      %83 = llvm.mul %82, %80 : !llvm.i32
      %84 = llvm.add %83, %81 : !llvm.i32
      %85 = llvm.getelementptr %69[%84] : (!llvm.ptr<half, 3>, !llvm.i32) -> !llvm.ptr<half, 3>
      %86 = llvm.bitcast %85 : !llvm.ptr<half, 3> to !llvm.ptr<i32, 3>
      %87 = nvvm.wmma.load %86, %82 {layout = "RowMajor", operand = "AOp", srcMemSpace = 3 : ui8, stride = 16 : ui16, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : !llvm.ptr<i32, 3>, !llvm.i32 -> !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %88 = llvm.bitcast %76 : !llvm.ptr<vec<2 x half>> to !llvm.ptr<i32>
      %89 = llvm.extractvalue %87[0 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %90 = llvm.bitcast %89 : !llvm.vec<2 x half> to !llvm.i32
      %91 = llvm.mlir.constant(0 : ui32) : !llvm.i32
      %92 = llvm.getelementptr %88[%91] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32, 3>
      llvm.store %90, %92 : !llvm.ptr<i32, 3>
      %93 = llvm.extractvalue %87[1 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %94 = llvm.bitcast %93 : !llvm.vec<2 x half> to !llvm.i32
      %95 = llvm.mlir.constant(1 : ui32) : !llvm.i32
      %96 = llvm.getelementptr %88[%95] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32, 3>
      llvm.store %94, %96 : !llvm.ptr<i32, 3>
      %97 = llvm.extractvalue %87[2 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %98 = llvm.bitcast %97 : !llvm.vec<2 x half> to !llvm.i32
      %99 = llvm.mlir.constant(2 : ui32) : !llvm.i32
      %100 = llvm.getelementptr %88[%99] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32, 3>
      llvm.store %98, %100 : !llvm.ptr<i32, 3>
      %101 = llvm.extractvalue %87[3 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %102 = llvm.bitcast %101 : !llvm.vec<2 x half> to !llvm.i32
      %103 = llvm.mlir.constant(3 : ui32) : !llvm.i32
      %104 = llvm.getelementptr %88[%103] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32, 3>
      llvm.store %102, %104 : !llvm.ptr<i32, 3>
      %105 = llvm.extractvalue %87[4 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %106 = llvm.bitcast %105 : !llvm.vec<2 x half> to !llvm.i32
      %107 = llvm.mlir.constant(4 : ui32) : !llvm.i32
      %108 = llvm.getelementptr %88[%107] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32, 3>
      llvm.store %106, %108 : !llvm.ptr<i32, 3>
      %109 = llvm.extractvalue %87[5 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %110 = llvm.bitcast %109 : !llvm.vec<2 x half> to !llvm.i32
      %111 = llvm.mlir.constant(5 : ui32) : !llvm.i32
      %112 = llvm.getelementptr %88[%111] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32, 3>
      llvm.store %110, %112 : !llvm.ptr<i32, 3>
      %113 = llvm.extractvalue %87[6 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %114 = llvm.bitcast %113 : !llvm.vec<2 x half> to !llvm.i32
      %115 = llvm.mlir.constant(6 : ui32) : !llvm.i32
      %116 = llvm.getelementptr %88[%115] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32, 3>
      llvm.store %114, %116 : !llvm.ptr<i32, 3>
      %117 = llvm.extractvalue %87[7 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %118 = llvm.bitcast %117 : !llvm.vec<2 x half> to !llvm.i32
      %119 = llvm.mlir.constant(7 : ui32) : !llvm.i32
      %120 = llvm.getelementptr %88[%119] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32, 3>
      llvm.store %118, %120 : !llvm.ptr<i32, 3>
      %121 = llvm.extractvalue %28[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %122 = llvm.extractvalue %28[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %123 = llvm.extractvalue %28[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %124 = llvm.extractvalue %28[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %125 = llvm.extractvalue %28[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %126 = llvm.extractvalue %41[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %127 = llvm.extractvalue %41[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %128 = llvm.extractvalue %41[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %129 = llvm.extractvalue %41[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %130 = llvm.extractvalue %41[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %131 = llvm.extractvalue %54[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %132 = llvm.extractvalue %54[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %133 = llvm.extractvalue %54[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %134 = llvm.extractvalue %54[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %135 = llvm.extractvalue %54[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %136 = llvm.extractvalue %67[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %137 = llvm.extractvalue %67[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %138 = llvm.extractvalue %67[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %139 = llvm.extractvalue %67[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %140 = llvm.extractvalue %67[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %141 = llvm.mlir.constant(0 : ui32) : !llvm.i32
      %142 = llvm.getelementptr %122[%141] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %143 = llvm.load %142 : !llvm.ptr<vec<2 x half>>
      %144 = llvm.mlir.constant(1 : ui32) : !llvm.i32
      %145 = llvm.getelementptr %122[%144] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %146 = llvm.load %145 : !llvm.ptr<vec<2 x half>>
      %147 = llvm.mlir.constant(2 : ui32) : !llvm.i32
      %148 = llvm.getelementptr %122[%147] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %149 = llvm.load %148 : !llvm.ptr<vec<2 x half>>
      %150 = llvm.mlir.constant(3 : ui32) : !llvm.i32
      %151 = llvm.getelementptr %122[%150] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %152 = llvm.load %151 : !llvm.ptr<vec<2 x half>>
      %153 = llvm.mlir.constant(4 : ui32) : !llvm.i32
      %154 = llvm.getelementptr %122[%153] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %155 = llvm.load %154 : !llvm.ptr<vec<2 x half>>
      %156 = llvm.mlir.constant(5 : ui32) : !llvm.i32
      %157 = llvm.getelementptr %122[%156] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %158 = llvm.load %157 : !llvm.ptr<vec<2 x half>>
      %159 = llvm.mlir.constant(6 : ui32) : !llvm.i32
      %160 = llvm.getelementptr %122[%159] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %161 = llvm.load %160 : !llvm.ptr<vec<2 x half>>
      %162 = llvm.mlir.constant(7 : ui32) : !llvm.i32
      %163 = llvm.getelementptr %122[%162] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %164 = llvm.load %163 : !llvm.ptr<vec<2 x half>>
      %165 = llvm.mlir.constant(0 : ui32) : !llvm.i32
      %166 = llvm.getelementptr %122[%165] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %167 = llvm.load %166 : !llvm.ptr<vec<2 x half>>
      %168 = llvm.mlir.constant(1 : ui32) : !llvm.i32
      %169 = llvm.getelementptr %122[%168] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %170 = llvm.load %169 : !llvm.ptr<vec<2 x half>>
      %171 = llvm.mlir.constant(2 : ui32) : !llvm.i32
      %172 = llvm.getelementptr %122[%171] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %173 = llvm.load %172 : !llvm.ptr<vec<2 x half>>
      %174 = llvm.mlir.constant(3 : ui32) : !llvm.i32
      %175 = llvm.getelementptr %122[%174] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %176 = llvm.load %175 : !llvm.ptr<vec<2 x half>>
      %177 = llvm.mlir.constant(4 : ui32) : !llvm.i32
      %178 = llvm.getelementptr %122[%177] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %179 = llvm.load %178 : !llvm.ptr<vec<2 x half>>
      %180 = llvm.mlir.constant(5 : ui32) : !llvm.i32
      %181 = llvm.getelementptr %122[%180] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %182 = llvm.load %181 : !llvm.ptr<vec<2 x half>>
      %183 = llvm.mlir.constant(6 : ui32) : !llvm.i32
      %184 = llvm.getelementptr %122[%183] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %185 = llvm.load %184 : !llvm.ptr<vec<2 x half>>
      %186 = llvm.mlir.constant(7 : ui32) : !llvm.i32
      %187 = llvm.getelementptr %122[%186] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %188 = llvm.load %187 : !llvm.ptr<vec<2 x half>>
      %189 = llvm.mlir.constant(0 : ui32) : !llvm.i32
      %190 = llvm.getelementptr %122[%189] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %191 = llvm.load %190 : !llvm.ptr<vec<2 x half>>
      %192 = llvm.mlir.constant(1 : ui32) : !llvm.i32
      %193 = llvm.getelementptr %122[%192] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %194 = llvm.load %193 : !llvm.ptr<vec<2 x half>>
      %195 = llvm.mlir.constant(2 : ui32) : !llvm.i32
      %196 = llvm.getelementptr %122[%195] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %197 = llvm.load %196 : !llvm.ptr<vec<2 x half>>
      %198 = llvm.mlir.constant(3 : ui32) : !llvm.i32
      %199 = llvm.getelementptr %122[%198] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %200 = llvm.load %199 : !llvm.ptr<vec<2 x half>>
      %201 = nvvm.wmma.mma %143, %146, %149, %152, %155, %158, %161, %164, %167, %170, %173, %176, %179, %182, %185, %188, %191, %194, %197, %200 {wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half> -> !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %202 = llvm.bitcast %137 : !llvm.ptr<vec<2 x half>> to !llvm.ptr<i32>
      %203 = llvm.extractvalue %201[0 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %204 = llvm.bitcast %203 : !llvm.vec<2 x half> to !llvm.i32
      %205 = llvm.mlir.constant(0 : ui32) : !llvm.i32
      %206 = llvm.getelementptr %202[%205] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %204, %206 : !llvm.ptr<i32>
      %207 = llvm.extractvalue %201[1 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %208 = llvm.bitcast %207 : !llvm.vec<2 x half> to !llvm.i32
      %209 = llvm.mlir.constant(1 : ui32) : !llvm.i32
      %210 = llvm.getelementptr %202[%209] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %208, %210 : !llvm.ptr<i32>
      %211 = llvm.extractvalue %201[2 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %212 = llvm.bitcast %211 : !llvm.vec<2 x half> to !llvm.i32
      %213 = llvm.mlir.constant(2 : ui32) : !llvm.i32
      %214 = llvm.getelementptr %202[%213] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %212, %214 : !llvm.ptr<i32>
      %215 = llvm.extractvalue %201[3 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %216 = llvm.bitcast %215 : !llvm.vec<2 x half> to !llvm.i32
      %217 = llvm.mlir.constant(3 : ui32) : !llvm.i32
      %218 = llvm.getelementptr %202[%217] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %216, %218 : !llvm.ptr<i32>
      %219 = llvm.extractvalue %67[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %220 = llvm.extractvalue %67[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %221 = llvm.extractvalue %67[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %222 = llvm.extractvalue %67[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %223 = llvm.extractvalue %67[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %224 = llvm.extractvalue %15[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %225 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %226 = llvm.extractvalue %15[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %227 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %228 = llvm.extractvalue %15[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %229 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %230 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %231 = llvm.mlir.constant(16 : index) : !llvm.i32
      %232 = llvm.mlir.constant(16 : index) : !llvm.i32
      %233 = llvm.mlir.constant(16 : ui16) : !llvm.i32
      %234 = llvm.mul %233, %231 : !llvm.i32
      %235 = llvm.add %234, %232 : !llvm.i32
      %236 = llvm.getelementptr %225[%235] : (!llvm.ptr<half, 3>, !llvm.i32) -> !llvm.ptr<half, 3>
      %237 = llvm.bitcast %236 : !llvm.ptr<half, 3> to !llvm.ptr<i32, 3>
      %238 = llvm.mlir.constant(0 : ui32) : !llvm.i32
      %239 = llvm.getelementptr %220[%238] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %240 = llvm.load %239 : !llvm.ptr<vec<2 x half>>
      %241 = llvm.mlir.constant(1 : ui32) : !llvm.i32
      %242 = llvm.getelementptr %220[%241] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %243 = llvm.load %242 : !llvm.ptr<vec<2 x half>>
      %244 = llvm.mlir.constant(2 : ui32) : !llvm.i32
      %245 = llvm.getelementptr %220[%244] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %246 = llvm.load %245 : !llvm.ptr<vec<2 x half>>
      %247 = llvm.mlir.constant(3 : ui32) : !llvm.i32
      %248 = llvm.getelementptr %220[%247] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %249 = llvm.load %248 : !llvm.ptr<vec<2 x half>>
      nvvm.wmma.store %237, %240, %243, %246, %249, %233 {layout = "RowMajor", srcMemSpace = 3 : ui8, stride = 16 : ui16, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : !llvm.ptr<i32, 3>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.i32
      llvm.return
    }
  }
}

