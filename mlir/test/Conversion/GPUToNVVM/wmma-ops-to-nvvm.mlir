// RUN: mlir-opt --convert-gpu-to-nvvm --split-input-file %s | FileCheck %s

gpu.module @test_module {

  // CHECK-LABEL: func @gpu_wmma_load_op()
  func @gpu_wmma_load_op() -> () {
    %wg = alloca() : memref<32x32xf16, 3>
    %A = alloca() : memref<8xvector<2xf16>, 5>
 
    gpu.subgroup_mma_load_matrix %wg, %A {operand = "AOp", srcOffsetJ = 16 : i64, srcOffsetI = 16 : i64, ldm = 32 : i64, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : memref<32x32xf16, 3>, memref<8xvector<2xf16>, 5>

    // CHECK: %{{.*}} = llvm.mlir.constant(32 : index) : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(32 : index) : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(1024 : index) : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.mlir.null : !llvm.ptr<half, 3>
    // CHECK-NEXT: %{{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<half, 3>, !llvm.i64) -> !llvm.ptr<half, 3>
    // CHECK-NEXT: %{{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<half, 3> to !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.alloca {{.*}} x !llvm.half : (!llvm.i64) -> !llvm.ptr<half, 3>
    // CHECK-NEXT: %{{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(8 : index) : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.mlir.null : !llvm.ptr<vec<2 x half>, 5>
    // CHECK-NEXT: %{{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<vec<2 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
    // CHECK-NEXT: %{{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<vec<2 x half>, 5> to !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.alloca {{.*}} x !llvm.vec<2 x half> : (!llvm.i64) -> !llvm.ptr<vec<2 x half>, 5>
    // CHECK-NEXT: %{{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %[[BASE:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %[[OFFSETT:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.extractvalue {{.*}}[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.extractvalue {{.*}}[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %[[STOREADDR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<vec<2 x half>>, ptr<vec<2 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %[[OFFI:.*]] = llvm.mlir.constant(16 : i64) : !llvm.i64
    // CHECK-NEXT: %[[OFFJ:.*]] = llvm.mlir.constant(16 : i64) : !llvm.i64
    // CHECK-NEXT: %[[LDM:.*]] = llvm.mlir.constant(32 : i64) : !llvm.i64
    // CHECK-NEXT: %[[ILDM:.*]] = llvm.mul %[[LDM]], %[[OFFI]] : !llvm.i64
    // CHECK-NEXT: %[[IJLDM:.*]] = llvm.add %[[ILDM]], %[[OFFJ]] : !llvm.i64
    // CHECK-NEXT: %[[IJOLDM:.*]] = llvm.add %[[IJLDM]], %[[OFFSETT]] : !llvm.i64
    // CHECK-NEXT: %[[NBASE:.*]] = llvm.getelementptr %[[BASE]][%[[IJOLDM]]] : (!llvm.ptr<half, 3>, !llvm.i64) -> !llvm.ptr<half, 3>
    // CHECK-NEXT: %[[CBASE:.*]] = llvm.bitcast %[[NBASE]] : !llvm.ptr<half, 3> to !llvm.ptr<i32, 3>
    // CHECK-NEXT: %[[STRIDE:.*]] = llvm.mlir.constant(32 : i64) : !llvm.i32
    // CHECK-NEXT: %[[FRAG:.*]] = nvvm.wmma.load %[[CBASE]], %[[STRIDE]] {ldm = 32 : i64, operand = "AOp", wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : !llvm.ptr<i32, 3>, !llvm.i32 -> !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    // CHECK-NEXT: %[[CSTOREADDR:.*]] = llvm.bitcast %[[STOREADDR]] : !llvm.ptr<vec<2 x half>> to !llvm.ptr<i32>
    // CHECK-NEXT: %[[ST0:.*]] = llvm.extractvalue %[[FRAG]][0 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    // CHECK-NEXT: %[[CST0:.*]] = llvm.bitcast %[[ST0]] : !llvm.vec<2 x half> to !llvm.i32
    // CHECK-NEXT: %[[OFF0:.*]] = llvm.mlir.constant(0 : ui32) : !llvm.i32
    // CHECK-NEXT: %[[FADDR0:.*]] = llvm.getelementptr %[[CSTOREADDR]][%[[OFF0]]] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    // CHECK-NEXT: llvm.store %[[CST0]], %[[FADDR0]] : !llvm.ptr<i32>
    // CHECK-NEXT: %[[ST1:.*]] = llvm.extractvalue %[[FRAG]][1 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    // CHECK-NEXT: %[[CST1:.*]] = llvm.bitcast %[[ST1]] : !llvm.vec<2 x half> to !llvm.i32
    // CHECK-NEXT: %[[OFF1:.*]] = llvm.mlir.constant(1 : ui32) : !llvm.i32
    // CHECK-NEXT: %[[FADDR1:.*]] = llvm.getelementptr %[[CSTOREADDR]][%[[OFF1]]] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    // CHECK-NEXT: llvm.store %[[CST1]], %[[FADDR1]] : !llvm.ptr<i32>
    // CHECK-NEXT: %[[ST2:.*]] = llvm.extractvalue %[[FRAG]][2 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    // CHECK-NEXT: %[[CST2:.*]] = llvm.bitcast %[[ST2]] : !llvm.vec<2 x half> to !llvm.i32
    // CHECK-NEXT: %[[OFF2:.*]] = llvm.mlir.constant(2 : ui32) : !llvm.i32
    // CHECK-NEXT: %[[FADDR2:.*]] = llvm.getelementptr %[[CSTOREADDR]][%[[OFF2]]] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    // CHECK-NEXT: llvm.store %[[CST2]], %[[FADDR2]] : !llvm.ptr<i32>
    // CHECK-NEXT: %[[ST3:.*]] = llvm.extractvalue %[[FRAG]][3 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    // CHECK-NEXT: %[[CST3:.*]] = llvm.bitcast %[[ST3]] : !llvm.vec<2 x half> to !llvm.i32
    // CHECK-NEXT: %[[OFF3:.*]] = llvm.mlir.constant(3 : ui32) : !llvm.i32
    // CHECK-NEXT: %[[FADDR3:.*]] = llvm.getelementptr %[[CSTOREADDR]][%[[OFF3]]] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    // CHECK-NEXT: llvm.store %[[CST3]], %[[FADDR3]] : !llvm.ptr<i32>
    // CHECK-NEXT: %[[ST4:.*]] = llvm.extractvalue %[[FRAG]][4 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    // CHECK-NEXT: %[[CST4:.*]] = llvm.bitcast %[[ST4]] : !llvm.vec<2 x half> to !llvm.i32
    // CHECK-NEXT: %[[OFF4:.*]] = llvm.mlir.constant(4 : ui32) : !llvm.i32
    // CHECK-NEXT: %[[FADDR4:.*]] = llvm.getelementptr %[[CSTOREADDR]][%[[OFF4]]] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    // CHECK-NEXT: llvm.store %[[CST4]], %[[FADDR4]] : !llvm.ptr<i32>
    // CHECK-NEXT: %[[ST5:.*]] = llvm.extractvalue %[[FRAG]][5 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    // CHECK-NEXT: %[[CST5:.*]] = llvm.bitcast %[[ST5]] : !llvm.vec<2 x half> to !llvm.i32
    // CHECK-NEXT: %[[OFF5:.*]] = llvm.mlir.constant(5 : ui32) : !llvm.i32
    // CHECK-NEXT: %[[FADDR5:.*]] = llvm.getelementptr %[[CSTOREADDR]][%[[OFF5]]] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    // CHECK-NEXT: llvm.store %[[CST5]], %[[FADDR5]] : !llvm.ptr<i32>
    // CHECK-NEXT: %[[ST6:.*]] = llvm.extractvalue %[[FRAG]][6 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    // CHECK-NEXT: %[[CST6:.*]] = llvm.bitcast %[[ST6]] : !llvm.vec<2 x half> to !llvm.i32
    // CHECK-NEXT: %[[OFF6:.*]] = llvm.mlir.constant(6 : ui32) : !llvm.i32
    // CHECK-NEXT: %[[FADDR6:.*]] = llvm.getelementptr %[[CSTOREADDR]][%[[OFF6]]] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    // CHECK-NEXT: llvm.store %[[CST6]], %[[FADDR6]] : !llvm.ptr<i32>
    // CHECK-NEXT: %[[ST7:.*]] = llvm.extractvalue %[[FRAG]][7 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    // CHECK-NEXT: %[[CST7:.*]] = llvm.bitcast %[[ST7]] : !llvm.vec<2 x half> to !llvm.i32
    // CHECK-NEXT: %[[OFF7:.*]] = llvm.mlir.constant(7 : ui32) : !llvm.i32
    // CHECK-NEXT: %[[FADDR7:.*]] = llvm.getelementptr %[[CSTOREADDR]][%[[OFF7]]] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    // CHECK-NEXT: llvm.store %[[CST7]], %[[FADDR7]] : !llvm.ptr<i32>
    // CHECK-NEXT: llvm.return
    return
  }
}

// -----

gpu.module @test_module {

  // CHECK-LABEL: func @gpu_wmma_store_op()
  func @gpu_wmma_store_op() -> () {
    %sg = alloca() : memref<32x32xf16, 3>
    %D = alloca() : memref<4xvector<2xf16>, 5>
    gpu.subgroup_mma_store_matrix %D, %sg {dstOffsetJ = 16 : i64, dstOffsetI = 16 : i64, ldm = 32 : i64, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : memref<4xvector<2xf16>, 5>, memref<32x32xf16, 3>

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
    // CHECK-NEXT: %[[OFFSETTT:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %[[OFFI:.*]] = llvm.mlir.constant(16 : i64) : !llvm.i64
    // CHECK-NEXT: %[[OFFJ:.*]] = llvm.mlir.constant(16 : i64) : !llvm.i64
    // CHECK-NEXT: %[[LDM:.*]] = llvm.mlir.constant(32 : i64) : !llvm.i64
    // CHECK-NEXT: %[[OILDM:.*]] = llvm.mul %[[LDM]], %[[OFFI]] : !llvm.i64
    // CHECK-NEXT: %[[OIJLDM:.*]] = llvm.add %[[OILDM]], %[[OFFJ]] : !llvm.i64
    // CHECK-NEXT: %[[TOFFSET:.*]] = llvm.add %[[OIJLDM]], %[[OFFSETTT]] : !llvm.i64
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

// -----

gpu.module @test_module {

  // CHECK-LABEL: func @gpu_wmma_mma_op()
  func @gpu_wmma_mma_op() -> () {
    %A = alloca() : memref<8xvector<2xf16>, 5>
    %B = alloca() : memref<8xvector<2xf16>, 5>
    %C = alloca() : memref<4xvector<2xf16>, 5>
    %D = alloca() : memref<4xvector<2xf16>, 5>

    gpu.subgroup_mma_compute %A, %B, %C, %D {ldm = 32 : ui16, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : memref<8xvector<2xf16>, 5>, memref<8xvector<2xf16>, 5>, memref<4xvector<2xf16>, 5>, memref<4xvector<2xf16>, 5>

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
