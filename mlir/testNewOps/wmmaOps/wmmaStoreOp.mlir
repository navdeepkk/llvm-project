gpu.module @test_module {

  // CHECK-LABEL: func @gpu_wmma_ops()
  func @gpu_wmma_ops() -> () {
    %sg = alloca() {alignment = 32} : memref<32x32xf16, 3>
    %D = alloca() : memref<1xvector<8xf16>, 5>
    %i = constant 16 : i64
    %j = constant 16 : i64
    %c0 = constant 0 : i64
    gpu.subgroup_mma_store_matrix %D[%c0], %sg[%i,%j] {ldm = 32 : i64} : memref<1xvector<8xf16>, 5>, memref<32x32xf16, 3>
    //gpu.subgroup_mma_store_matrix %D, %sg {dstOffsetJ = 16 : i64, dstOffsetI = 16 : i64, ldm = 32 : i64, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : memref<4xvector<2xf16>, 5>, memref<32x32xf16, 3>

    // CHECK: {{.*}} = llvm.mlir.constant(32 : index) : !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(32 : index) : !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(1024 : index) : !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.mlir.null : !llvm.ptr<half, 3>
    // CHECK-NEXT: {{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<half, 3>, !llvm.i64) -> !llvm.ptr<half, 3>
    // CHECK-NEXT: {{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<half, 3> to !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.alloca {{.*}} x !llvm.half : (!llvm.i64) -> !llvm.ptr<half, 3>
    // CHECK-NEXT: {{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(4 : index) : !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.mlir.null : !llvm.ptr<vec<2 x half>, 5>
    // CHECK-NEXT: {{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<vec<2 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
    // CHECK-NEXT: {{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<vec<2 x half>, 5> to !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.alloca {{.*}} x !llvm.vec<2 x half> : (!llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
    // CHECK-NEXT: {{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %[[SRCADDR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %[[BASEADDR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %[[OFFSET:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %[[OFFI:.*]] = llvm.mlir.constant(16 : i64) : !llvm.i64
    // CHECK-NEXT: %[[OFFJ:.*]] = llvm.mlir.constant(16 : i64) : !llvm.i64
    // CHECK-NEXT: %[[LDM:.*]] = llvm.mlir.constant(32 : i64) : !llvm.i64
    // CHECK-NEXT: %[[OILDM:.*]] = llvm.mul %[[LDM]], %[[OFFI]] : !llvm.i64
    // CHECK-NEXT: %[[OIJLDM:.*]] = llvm.add %[[OILDM]], %[[OFFJ]] : !llvm.i64
    // CHECK-NEXT: %[[TOFFSET:.*]] = llvm.add %[[OIJLDM]], %[[OFFSET]] : !llvm.i64
    // CHECK-NEXT: %[[LADDR:.*]] = llvm.getelementptr %[[BASEADDR]][%[[TOFFSET]]] : (!llvm.ptr<half, 3>, !llvm.i64) -> !llvm.ptr<half, 3>
    // CHECK-NEXT: %[[CADDR:.*]] = llvm.bitcast %[[LADDR]] : !llvm.ptr<half, 3> to !llvm.ptr<i32, 3>
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(0 : ui32) : !llvm.i32
    // CHECK-NEXT: %[[ADDR0:.*]] = llvm.getelementptr %[[SRCADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    // CHECK-NEXT: %[[EL0:.*]] = llvm.load %[[ADDR0]] : !llvm.ptr<vec<2 x half>>
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(1 : ui32) : !llvm.i32
    // CHECK-NEXT: %[[ADDR1:.*]] = llvm.getelementptr %[[SRCADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    // CHECK-NEXT: %[[EL1:.*]] = llvm.load %[[ADDR1]] : !llvm.ptr<vec<2 x half>>
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(2 : ui32) : !llvm.i32
    // CHECK-NEXT: %[[ADDR2:.*]] = llvm.getelementptr %[[SRCADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    // CHECK-NEXT: %[[EL2:.*]] = llvm.load %[[ADDR2]] : !llvm.ptr<vec<2 x half>>
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(3 : ui32) : !llvm.i32
    // CHECK-NEXT: %[[ADDR3:.*]] = llvm.getelementptr %[[SRCADDR]][{{.*}}] : (!llvm.ptr<vec<2 x half>>, !llvm.i32) -> !llvm.ptr<vec<2 x half>>
    // CHECK-NEXT: %[[EL3:.*]] = llvm.load %[[ADDR3]] : !llvm.ptr<vec<2 x half>>
    // CHECK-NEXT: %[[STRIDE:.*]] = llvm.mlir.constant(32 : i64) : !llvm.i32
    // CHECK-NEXT: nvvm.wmma.store %[[CADDR]], %[[EL0]], %[[EL1]], %[[EL2]], %[[EL3]], %[[STRIDE]] {ldm = 32 : i64, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : !llvm.ptr<i32, 3>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.i32
    return
  }
}
