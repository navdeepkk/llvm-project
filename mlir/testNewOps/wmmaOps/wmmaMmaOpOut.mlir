module  {
  gpu.module @test_module {
    llvm.func @gpu_wmma_ops() {
      %0 = llvm.mlir.constant(8 : index) : !llvm.i64
      %1 = llvm.mlir.constant(1 : index) : !llvm.i64
      %2 = llvm.mlir.null : !llvm.ptr<vec<2 x half>, 5>
      %3 = llvm.getelementptr %2[%0] : (!llvm.ptr<vec<2 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
      %4 = llvm.ptrtoint %3 : !llvm.ptr<vec<2 x half>, 5> to !llvm.i64
      %5 = llvm.alloca %4 x !llvm.vec<2 x half> : (!llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
      %6 = llvm.mlir.undef : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %7 = llvm.insertvalue %5, %6[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %8 = llvm.insertvalue %5, %7[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %9 = llvm.mlir.constant(0 : index) : !llvm.i64
      %10 = llvm.insertvalue %9, %8[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %11 = llvm.insertvalue %0, %10[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %12 = llvm.insertvalue %1, %11[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %13 = llvm.mlir.constant(8 : index) : !llvm.i64
      %14 = llvm.mlir.constant(1 : index) : !llvm.i64
      %15 = llvm.mlir.null : !llvm.ptr<vec<2 x half>, 5>
      %16 = llvm.getelementptr %15[%13] : (!llvm.ptr<vec<2 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
      %17 = llvm.ptrtoint %16 : !llvm.ptr<vec<2 x half>, 5> to !llvm.i64
      %18 = llvm.alloca %17 x !llvm.vec<2 x half> : (!llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
      %19 = llvm.mlir.undef : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %20 = llvm.insertvalue %18, %19[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %21 = llvm.insertvalue %18, %20[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %22 = llvm.mlir.constant(0 : index) : !llvm.i64
      %23 = llvm.insertvalue %22, %21[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %24 = llvm.insertvalue %13, %23[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %25 = llvm.insertvalue %14, %24[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %26 = llvm.mlir.constant(4 : index) : !llvm.i64
      %27 = llvm.mlir.constant(1 : index) : !llvm.i64
      %28 = llvm.mlir.null : !llvm.ptr<vec<2 x half>, 5>
      %29 = llvm.getelementptr %28[%26] : (!llvm.ptr<vec<2 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
      %30 = llvm.ptrtoint %29 : !llvm.ptr<vec<2 x half>, 5> to !llvm.i64
      %31 = llvm.alloca %30 x !llvm.vec<2 x half> : (!llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
      %32 = llvm.mlir.undef : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %33 = llvm.insertvalue %31, %32[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %34 = llvm.insertvalue %31, %33[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %35 = llvm.mlir.constant(0 : index) : !llvm.i64
      %36 = llvm.insertvalue %35, %34[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %37 = llvm.insertvalue %26, %36[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %38 = llvm.insertvalue %27, %37[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %39 = llvm.mlir.constant(4 : index) : !llvm.i64
      %40 = llvm.mlir.constant(1 : index) : !llvm.i64
      %41 = llvm.mlir.null : !llvm.ptr<vec<2 x half>, 5>
      %42 = llvm.getelementptr %41[%39] : (!llvm.ptr<vec<2 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
      %43 = llvm.ptrtoint %42 : !llvm.ptr<vec<2 x half>, 5> to !llvm.i64
      %44 = llvm.alloca %43 x !llvm.vec<2 x half> : (!llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
      %45 = llvm.mlir.undef : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %46 = llvm.insertvalue %44, %45[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %47 = llvm.insertvalue %44, %46[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %48 = llvm.mlir.constant(0 : index) : !llvm.i64
      %49 = llvm.insertvalue %48, %47[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %50 = llvm.insertvalue %39, %49[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %51 = llvm.insertvalue %40, %50[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %52 = llvm.extractvalue %12[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %53 = llvm.extractvalue %12[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %54 = llvm.extractvalue %12[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %55 = llvm.extractvalue %12[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %56 = llvm.extractvalue %12[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %57 = llvm.extractvalue %25[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %58 = llvm.extractvalue %25[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %59 = llvm.extractvalue %25[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %60 = llvm.extractvalue %25[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %61 = llvm.extractvalue %25[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %62 = llvm.extractvalue %38[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %63 = llvm.extractvalue %38[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %64 = llvm.extractvalue %38[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %65 = llvm.extractvalue %38[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %66 = llvm.extractvalue %38[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %67 = llvm.extractvalue %51[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %68 = llvm.extractvalue %51[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %69 = llvm.extractvalue %51[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %70 = llvm.extractvalue %51[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %71 = llvm.extractvalue %51[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %72 = llvm.mlir.constant(0 : ui32) : !llvm.i32
      %73 = llvm.getelementptr %53[%72] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %74 = llvm.load %73 : !llvm.ptr<vec<2 x half>>
      %75 = llvm.mlir.constant(1 : ui32) : !llvm.i32
      %76 = llvm.getelementptr %53[%75] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %77 = llvm.load %76 : !llvm.ptr<vec<2 x half>>
      %78 = llvm.mlir.constant(2 : ui32) : !llvm.i32
      %79 = llvm.getelementptr %53[%78] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %80 = llvm.load %79 : !llvm.ptr<vec<2 x half>>
      %81 = llvm.mlir.constant(3 : ui32) : !llvm.i32
      %82 = llvm.getelementptr %53[%81] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %83 = llvm.load %82 : !llvm.ptr<vec<2 x half>>
      %84 = llvm.mlir.constant(4 : ui32) : !llvm.i32
      %85 = llvm.getelementptr %53[%84] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %86 = llvm.load %85 : !llvm.ptr<vec<2 x half>>
      %87 = llvm.mlir.constant(5 : ui32) : !llvm.i32
      %88 = llvm.getelementptr %53[%87] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %89 = llvm.load %88 : !llvm.ptr<vec<2 x half>>
      %90 = llvm.mlir.constant(6 : ui32) : !llvm.i32
      %91 = llvm.getelementptr %53[%90] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %92 = llvm.load %91 : !llvm.ptr<vec<2 x half>>
      %93 = llvm.mlir.constant(7 : ui32) : !llvm.i32
      %94 = llvm.getelementptr %53[%93] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %95 = llvm.load %94 : !llvm.ptr<vec<2 x half>>
      %96 = llvm.mlir.constant(0 : ui32) : !llvm.i32
      %97 = llvm.getelementptr %58[%96] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %98 = llvm.load %97 : !llvm.ptr<vec<2 x half>>
      %99 = llvm.mlir.constant(1 : ui32) : !llvm.i32
      %100 = llvm.getelementptr %58[%99] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %101 = llvm.load %100 : !llvm.ptr<vec<2 x half>>
      %102 = llvm.mlir.constant(2 : ui32) : !llvm.i32
      %103 = llvm.getelementptr %58[%102] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %104 = llvm.load %103 : !llvm.ptr<vec<2 x half>>
      %105 = llvm.mlir.constant(3 : ui32) : !llvm.i32
      %106 = llvm.getelementptr %58[%105] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %107 = llvm.load %106 : !llvm.ptr<vec<2 x half>>
      %108 = llvm.mlir.constant(4 : ui32) : !llvm.i32
      %109 = llvm.getelementptr %58[%108] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %110 = llvm.load %109 : !llvm.ptr<vec<2 x half>>
      %111 = llvm.mlir.constant(5 : ui32) : !llvm.i32
      %112 = llvm.getelementptr %58[%111] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %113 = llvm.load %112 : !llvm.ptr<vec<2 x half>>
      %114 = llvm.mlir.constant(6 : ui32) : !llvm.i32
      %115 = llvm.getelementptr %58[%114] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %116 = llvm.load %115 : !llvm.ptr<vec<2 x half>>
      %117 = llvm.mlir.constant(7 : ui32) : !llvm.i32
      %118 = llvm.getelementptr %58[%117] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %119 = llvm.load %118 : !llvm.ptr<vec<2 x half>>
      %120 = llvm.mlir.constant(0 : ui32) : !llvm.i32
      %121 = llvm.getelementptr %63[%120] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %122 = llvm.load %121 : !llvm.ptr<vec<2 x half>>
      %123 = llvm.mlir.constant(1 : ui32) : !llvm.i32
      %124 = llvm.getelementptr %63[%123] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %125 = llvm.load %124 : !llvm.ptr<vec<2 x half>>
      %126 = llvm.mlir.constant(2 : ui32) : !llvm.i32
      %127 = llvm.getelementptr %63[%126] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %128 = llvm.load %127 : !llvm.ptr<vec<2 x half>>
      %129 = llvm.mlir.constant(3 : ui32) : !llvm.i32
      %130 = llvm.getelementptr %63[%129] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
      %131 = llvm.load %130 : !llvm.ptr<vec<2 x half>>
      %132 = nvvm.wmma.mma %74, %77, %80, %83, %86, %89, %92, %95, %98, %101, %104, %107, %110, %113, %116, %119, %122, %125, %128, %131 {wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half> -> !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %133 = llvm.bitcast %68 : !llvm.ptr<vec<2 x half>> to !llvm.ptr<i32>
      %134 = llvm.extractvalue %132[0 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %135 = llvm.bitcast %134 : !llvm.vec<2 x half> to !llvm.i32
      %136 = llvm.mlir.constant(0 : ui32) : !llvm.i32
      %137 = llvm.getelementptr %133[%136] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %135, %137 : !llvm.ptr<i32>
      %138 = llvm.extractvalue %132[1 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %139 = llvm.bitcast %138 : !llvm.vec<2 x half> to !llvm.i32
      %140 = llvm.mlir.constant(1 : ui32) : !llvm.i32
      %141 = llvm.getelementptr %133[%140] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %139, %141 : !llvm.ptr<i32>
      %142 = llvm.extractvalue %132[2 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %143 = llvm.bitcast %142 : !llvm.vec<2 x half> to !llvm.i32
      %144 = llvm.mlir.constant(2 : ui32) : !llvm.i32
      %145 = llvm.getelementptr %133[%144] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %143, %145 : !llvm.ptr<i32>
      %146 = llvm.extractvalue %132[3 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %147 = llvm.bitcast %146 : !llvm.vec<2 x half> to !llvm.i32
      %148 = llvm.mlir.constant(3 : ui32) : !llvm.i32
      %149 = llvm.getelementptr %133[%148] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %147, %149 : !llvm.ptr<i32>
      llvm.return
    }
    llvm.func @_mlir_ciface_gpu_wmma_ops() {
      llvm.call @gpu_wmma_ops() : () -> ()
      llvm.return
    }
  }
}

