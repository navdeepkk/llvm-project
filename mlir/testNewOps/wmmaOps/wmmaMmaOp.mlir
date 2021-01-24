gpu.module @test_module {

  // CHECK-LABEL: func @gpu_wmma_ops()
  func @gpu_wmma_ops() -> () {
    //%A = alloca() : memref<8xvector<2xf16>, 5>
    //%B = alloca() : memref<8xvector<2xf16>, 5>
    //%C = alloca() : memref<4xvector<2xf16>, 5>
    //%D = alloca() : memref<4xvector<2xf16>, 5>
    %A = alloca() : memref<1xvector<16xf16>, 5>
    %B = alloca() : memref<1xvector<16xf16>, 5>
    %C = alloca() : memref<1xvector<8xf16>, 5>
    %D = alloca() : memref<1xvector<8xf16>, 5>
    %c0 =  constant 0 : i64
    //gpu.subgroup_mma_compute %A, %B, %C, %D {ldm = 32 : ui16, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : memref<8xvector<2xf16>, 5>, memref<8xvector<2xf16>, 5>, memref<4xvector<2xf16>, 5>, memref<4xvector<2xf16>, 5>
    gpu.subgroup_mma_compute %A[%c0], %B[%c0], %C[%c0], %D[%c0] : memref<1xvector<16xf16>, 5>, memref<1xvector<16xf16>, 5>, memref<1xvector<8xf16>, 5>, memref<1xvector<8xf16>, 5>

    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(8 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.mlir.null : !llvm.ptr<vec<2 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<vec<2 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<vec<2 x half>, 5> to !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.alloca {{.*}} x !llvm.vec<2 x half> : (!llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(8 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.mlir.null : !llvm.ptr<vec<2 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<vec<2 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<vec<2 x half>, 5> to !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.alloca {{.*}} x !llvm.vec<2 x half> : (!llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(4 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.mlir.null : !llvm.ptr<vec<2 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<vec<2 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<vec<2 x half>, 5> to !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.alloca {{.*}} x !llvm.vec<2 x half> : (!llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(4 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.mlir.null : !llvm.ptr<vec<2 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<vec<2 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<vec<2 x half>, 5> to !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.alloca {{.*}} x !llvm.vec<2 x half> : (!llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:%[[AADDR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:%[[BADDR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:%[[CADDR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[A0:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[A1:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(2 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[A2:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(3 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[A3:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(4 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[A4:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(5 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[A5:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(6 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[A6:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(7 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[A7:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[B0:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[B1:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(2 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[B2:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(3 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[B3:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(4 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[B4:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(5 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[B5:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(6 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[B6:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(7 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[B7:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[CADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[C0:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[CADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[C1:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(2 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[CADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[C2:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(3 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[CADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:%[[C3:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<2 x half>>
    //CHECK-NEXT:{{.*}} = nvvm.wmma.mma %[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]], %[[A6]], %[[A7]], %[[B0]], %[[B1]], %[[B2]], %[[B3]], %[[B4]], %[[B5]], %[[B6]], %[[B7]], %[[C0]], %[[C1]], %[[C2]], %[[C3]] {wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half> -> !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    //CHECK-NEXT:{{.*}} = llvm.bitcast {{.*}} : !llvm.ptr<vec<2 x half>> to !llvm.ptr<i32>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[0 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    //CHECK-NEXT:%[[D0:.*]] = llvm.bitcast {{.*}} : !llvm.vec<2 x half> to !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : ui32) : !llvm.i32
    //CHECK-NEXT:%[[DADDR0:.*]] = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:llvm.store %[[D0]], %[[DADDR0]] : !llvm.ptr<i32>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[1 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    //CHECK-NEXT:%[[D1:.*]] = llvm.bitcast {{.*}} : !llvm.vec<2 x half> to !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : ui32) : !llvm.i32
    //CHECK-NEXT:%[[DADDR1:.*]] = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT: llvm.store %[[D1]], %[[DADDR1]] : !llvm.ptr<i32>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[2 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    //CHECK-NEXT:%[[D2:.*]] = llvm.bitcast {{.*}} : !llvm.vec<2 x half> to !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(2 : ui32) : !llvm.i32
    //CHECK-NEXT:%[[DADDR2:.*]] = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT: llvm.store %[[D2]], %[[DADDR2]] : !llvm.ptr<i32>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[3 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    //CHECK-NEXT:%[[D3:.*]] = llvm.bitcast {{.*}} : !llvm.vec<2 x half> to !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(3 : ui32) : !llvm.i32
    //CHECK-NEXT:%[[DADDR3:.*]] = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT: llvm.store %[[D3]], %[[DADDR3]] : !llvm.ptr<i32>
    //CHECK-NEXT:{{.*}} llvm.return
    
    return
  }
}
