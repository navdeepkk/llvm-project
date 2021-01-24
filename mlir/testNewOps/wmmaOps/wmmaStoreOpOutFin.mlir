module  {
  gpu.module @test_module {
    llvm.func @gpu_wmma_ops() {
      %0 = llvm.mlir.constant(16 : i64) : !llvm.i64
      %1 = llvm.mlir.constant(0 : i64) : !llvm.i64
      %2 = llvm.mlir.constant(32 : index) : !llvm.i64
      %3 = llvm.mlir.constant(32 : index) : !llvm.i64
      %4 = llvm.mlir.constant(1 : index) : !llvm.i64
      %5 = llvm.mlir.constant(1024 : index) : !llvm.i64
      %6 = llvm.mlir.null : !llvm.ptr<half, 3>
      %7 = llvm.getelementptr %6[%5] : (!llvm.ptr<half, 3>, !llvm.i64) -> !llvm.ptr<half, 3>
      %8 = llvm.ptrtoint %7 : !llvm.ptr<half, 3> to !llvm.i64
      %9 = llvm.alloca %8 x !llvm.half {alignment = 32 : i64} : (!llvm.i64) -> !llvm.ptr<half, 3>
      %10 = llvm.mlir.undef : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %11 = llvm.insertvalue %9, %10[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %12 = llvm.insertvalue %9, %11[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %13 = llvm.mlir.constant(0 : index) : !llvm.i64
      %14 = llvm.insertvalue %13, %12[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %15 = llvm.insertvalue %2, %14[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %16 = llvm.insertvalue %3, %15[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %17 = llvm.insertvalue %3, %16[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %18 = llvm.insertvalue %4, %17[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %19 = llvm.mlir.constant(1 : index) : !llvm.i64
      %20 = llvm.mlir.constant(1 : index) : !llvm.i64
      %21 = llvm.mlir.null : !llvm.ptr<vec<8 x half>, 5>
      %22 = llvm.getelementptr %21[%19] : (!llvm.ptr<vec<8 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<8 x half>, 5>
      %23 = llvm.ptrtoint %22 : !llvm.ptr<vec<8 x half>, 5> to !llvm.i64
      %24 = llvm.alloca %23 x !llvm.vec<8 x half> : (!llvm.i64) -> !llvm.ptr<vec<8 x half>, 5>
      %25 = llvm.mlir.undef : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %26 = llvm.insertvalue %24, %25[0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %27 = llvm.insertvalue %24, %26[1] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %28 = llvm.mlir.constant(0 : index) : !llvm.i64
      %29 = llvm.insertvalue %28, %27[2] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %30 = llvm.insertvalue %19, %29[3, 0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %31 = llvm.insertvalue %20, %30[4, 0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %32 = llvm.extractvalue %31[0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %33 = llvm.extractvalue %31[1] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %34 = llvm.extractvalue %31[2] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %35 = llvm.extractvalue %31[3, 0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %36 = llvm.extractvalue %31[4, 0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %37 = llvm.extractvalue %18[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %38 = llvm.extractvalue %18[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %39 = llvm.extractvalue %18[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %40 = llvm.extractvalue %18[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %41 = llvm.extractvalue %18[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %42 = llvm.extractvalue %18[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %43 = llvm.extractvalue %18[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %44 = llvm.mlir.constant(32 : i64) : !llvm.i64
      %45 = llvm.mul %44, %0 : !llvm.i64
      %46 = llvm.add %45, %0 : !llvm.i64
      %47 = llvm.add %46, %39 : !llvm.i64
      %48 = llvm.getelementptr %38[%47] : (!llvm.ptr<half, 3>, !llvm.i64) -> !llvm.ptr<half, 3>
      %49 = llvm.bitcast %48 : !llvm.ptr<half, 3> to !llvm.ptr<i32, 3>
      %50 = llvm.getelementptr %33[%1] : (!llvm.ptr<vec<8 x half>>, !llvm.i64) -> !llvm.ptr<vec<8 x half>>
      %51 = llvm.bitcast %50 : !llvm.ptr<vec<8 x half>> to !llvm.ptr<i32>
      %52 = llvm.mlir.constant(0 : ui32) : !llvm.i32
      %53 = llvm.getelementptr %51[%52] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      %54 = llvm.load %53 : !llvm.ptr<i32>
      %55 = llvm.bitcast %54 : !llvm.i32 to !llvm.vec<2 x half>
      %56 = llvm.mlir.constant(1 : ui32) : !llvm.i32
      %57 = llvm.getelementptr %51[%56] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      %58 = llvm.load %57 : !llvm.ptr<i32>
      %59 = llvm.bitcast %58 : !llvm.i32 to !llvm.vec<2 x half>
      %60 = llvm.mlir.constant(2 : ui32) : !llvm.i32
      %61 = llvm.getelementptr %51[%60] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      %62 = llvm.load %61 : !llvm.ptr<i32>
      %63 = llvm.bitcast %62 : !llvm.i32 to !llvm.vec<2 x half>
      %64 = llvm.mlir.constant(3 : ui32) : !llvm.i32
      %65 = llvm.getelementptr %51[%64] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      %66 = llvm.load %65 : !llvm.ptr<i32>
      %67 = llvm.bitcast %66 : !llvm.i32 to !llvm.vec<2 x half>
      %68 = llvm.mlir.constant(32 : i64) : !llvm.i32
      nvvm.wmma.m16n16k16.store %49, %55, %59, %63, %67, %68 : !llvm.ptr<i32, 3>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.i32
      llvm.return
    }
    llvm.func @_mlir_ciface_gpu_wmma_ops() {
      llvm.call @gpu_wmma_ops() : () -> ()
      llvm.return
    }
  }
}

