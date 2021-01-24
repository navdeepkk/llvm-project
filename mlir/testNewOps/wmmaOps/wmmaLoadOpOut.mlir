module  {
  gpu.module @test_module {
    llvm.func @gpu_wmma_ops() {
      %0 = llvm.mlir.constant(32 : index) : !llvm.i64
      %1 = llvm.mlir.constant(32 : index) : !llvm.i64
      %2 = llvm.mlir.constant(1 : index) : !llvm.i64
      %3 = llvm.mlir.constant(1024 : index) : !llvm.i64
      %4 = llvm.mlir.null : !llvm.ptr<half, 3>
      %5 = llvm.getelementptr %4[%3] : (!llvm.ptr<half, 3>, !llvm.i64) -> !llvm.ptr<half, 3>
      %6 = llvm.ptrtoint %5 : !llvm.ptr<half, 3> to !llvm.i64
      %7 = llvm.alloca %6 x !llvm.half : (!llvm.i64) -> !llvm.ptr<half, 3>
      %8 = llvm.mlir.undef : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %10 = llvm.insertvalue %7, %9[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %11 = llvm.mlir.constant(0 : index) : !llvm.i64
      %12 = llvm.insertvalue %11, %10[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %13 = llvm.insertvalue %0, %12[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.insertvalue %1, %13[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %15 = llvm.insertvalue %1, %14[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %16 = llvm.insertvalue %2, %15[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %17 = llvm.mlir.constant(8 : index) : !llvm.i64
      %18 = llvm.mlir.constant(1 : index) : !llvm.i64
      %19 = llvm.mlir.null : !llvm.ptr<vec<2 x half>, 5>
      %20 = llvm.getelementptr %19[%17] : (!llvm.ptr<vec<2 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
      %21 = llvm.ptrtoint %20 : !llvm.ptr<vec<2 x half>, 5> to !llvm.i64
      %22 = llvm.alloca %21 x !llvm.vec<2 x half> : (!llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
      %23 = llvm.mlir.undef : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %24 = llvm.insertvalue %22, %23[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %25 = llvm.insertvalue %22, %24[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %26 = llvm.mlir.constant(0 : index) : !llvm.i64
      %27 = llvm.insertvalue %26, %25[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %28 = llvm.insertvalue %17, %27[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %29 = llvm.insertvalue %18, %28[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %30 = llvm.extractvalue %16[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %31 = llvm.extractvalue %16[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %32 = llvm.extractvalue %16[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %33 = llvm.extractvalue %16[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %34 = llvm.extractvalue %16[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %35 = llvm.extractvalue %16[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %36 = llvm.extractvalue %16[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %37 = llvm.extractvalue %29[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %38 = llvm.extractvalue %29[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %39 = llvm.extractvalue %29[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %40 = llvm.extractvalue %29[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %41 = llvm.extractvalue %29[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %42 = llvm.mlir.constant(16 : i64) : !llvm.i64
      %43 = llvm.mlir.constant(16 : i64) : !llvm.i64
      %44 = llvm.mlir.constant(32 : i64) : !llvm.i64
      %45 = llvm.mul %44, %42 : !llvm.i64
      %46 = llvm.add %45, %43 : !llvm.i64
      %47 = llvm.add %46, %32 : !llvm.i64
      %48 = llvm.getelementptr %31[%47] : (!llvm.ptr<half, 3>, !llvm.i64) -> !llvm.ptr<half, 3>
      %49 = llvm.bitcast %48 : !llvm.ptr<half, 3> to !llvm.ptr<i32, 3>
      %50 = llvm.mlir.constant(32 : i64) : !llvm.i32
      %51 = nvvm.wmma.load %49, %50 {ldm = 32 : i64, operand = "AOp", wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : !llvm.ptr<i32, 3>, !llvm.i32 -> !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %52 = llvm.bitcast %38 : !llvm.ptr<vec<2 x half>> to !llvm.ptr<i32>
      %53 = llvm.extractvalue %51[0 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %54 = llvm.bitcast %53 : !llvm.vec<2 x half> to !llvm.i32
      %55 = llvm.mlir.constant(0 : ui32) : !llvm.i32
      %56 = llvm.getelementptr %52[%55] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %54, %56 : !llvm.ptr<i32>
      %57 = llvm.extractvalue %51[1 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %58 = llvm.bitcast %57 : !llvm.vec<2 x half> to !llvm.i32
      %59 = llvm.mlir.constant(1 : ui32) : !llvm.i32
      %60 = llvm.getelementptr %52[%59] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %58, %60 : !llvm.ptr<i32>
      %61 = llvm.extractvalue %51[2 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %62 = llvm.bitcast %61 : !llvm.vec<2 x half> to !llvm.i32
      %63 = llvm.mlir.constant(2 : ui32) : !llvm.i32
      %64 = llvm.getelementptr %52[%63] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %62, %64 : !llvm.ptr<i32>
      %65 = llvm.extractvalue %51[3 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %66 = llvm.bitcast %65 : !llvm.vec<2 x half> to !llvm.i32
      %67 = llvm.mlir.constant(3 : ui32) : !llvm.i32
      %68 = llvm.getelementptr %52[%67] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %66, %68 : !llvm.ptr<i32>
      %69 = llvm.extractvalue %51[4 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %70 = llvm.bitcast %69 : !llvm.vec<2 x half> to !llvm.i32
      %71 = llvm.mlir.constant(4 : ui32) : !llvm.i32
      %72 = llvm.getelementptr %52[%71] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %70, %72 : !llvm.ptr<i32>
      %73 = llvm.extractvalue %51[5 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %74 = llvm.bitcast %73 : !llvm.vec<2 x half> to !llvm.i32
      %75 = llvm.mlir.constant(5 : ui32) : !llvm.i32
      %76 = llvm.getelementptr %52[%75] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %74, %76 : !llvm.ptr<i32>
      %77 = llvm.extractvalue %51[6 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %78 = llvm.bitcast %77 : !llvm.vec<2 x half> to !llvm.i32
      %79 = llvm.mlir.constant(6 : ui32) : !llvm.i32
      %80 = llvm.getelementptr %52[%79] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %78, %80 : !llvm.ptr<i32>
      %81 = llvm.extractvalue %51[7 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %82 = llvm.bitcast %81 : !llvm.vec<2 x half> to !llvm.i32
      %83 = llvm.mlir.constant(7 : ui32) : !llvm.i32
      %84 = llvm.getelementptr %52[%83] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %82, %84 : !llvm.ptr<i32>
      llvm.return
    }
    llvm.func @_mlir_ciface_gpu_wmma_ops() {
      llvm.call @gpu_wmma_ops() : () -> ()
      llvm.return
    }
  }
}

