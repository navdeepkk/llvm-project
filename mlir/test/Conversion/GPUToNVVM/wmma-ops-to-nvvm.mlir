// RUN: mlir-opt --convert-gpu-to-nvvm --split-input-file %s | FileCheck %s

gpu.module @test_module {

  // CHECK-LABEL: func @gpu_wmma_load_op()
  func @gpu_wmma_load_op() -> () {
    %wg = alloca() {alignment = 32} : memref<32x32xf16, 3>
    %A = alloca() : memref<1xvector<16xf16>, 5>
    %i = constant 16 : i64
    %j = constant 16 : i64
    %c0 = constant 0 : i64
    gpu.subgroup_mma_load_matrix %wg[%i, %j], %A[%c0] {operand = "AOp", ldm = 32 : i64} : memref<32x32xf16, 3>, memref<1xvector<16xf16>, 5>

    // CHECK: %[[OFF:.*]] = llvm.mlir.constant(16 : i64) : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(32 : index) : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(32 : index) : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(1024 : index) : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.mlir.null : !llvm.ptr<half, 3>
    // CHECK-NEXT: %{{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<half, 3>, !llvm.i64) -> !llvm.ptr<half, 3>
    // CHECK-NEXT: %{{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<half, 3> to !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.alloca {{.*}} x !llvm.half {alignment = 32 : i64} : (!llvm.i64) -> !llvm.ptr<half, 3>
    // CHECK-NEXT: %{{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.mlir.null : !llvm.ptr<vec<16 x half>, 5>
    // CHECK-NEXT: %{{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<vec<16 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<16 x half>, 5>
    // CHECK-NEXT: %{{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<vec<16 x half>, 5> to !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.alloca {{.*}} x !llvm.vec<16 x half> : (!llvm.i64) -> !llvm.ptr<vec<16 x half>, 5>
    // CHECK-NEXT: %{{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %[[BASE:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %[[OFFSETT:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.extractvalue {{.*}}[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.extractvalue {{.*}}[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %[[STOREADDR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %{{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %[[LDM:.*]] = llvm.mlir.constant(32 : i64) : !llvm.i64
    // CHECK-NEXT: %[[ILDM:.*]] = llvm.mul %[[LDM]], %[[OFF]] : !llvm.i64
    // CHECK-NEXT: %[[IJLDM:.*]] = llvm.add %[[ILDM]], %[[OFF]] : !llvm.i64
    // CHECK-NEXT: %[[IJOLDM:.*]] = llvm.add %[[IJLDM]], %[[OFFSETT]] : !llvm.i64
    // CHECK-NEXT: %[[NBASE:.*]] = llvm.getelementptr %[[BASE]][%[[IJOLDM]]] : (!llvm.ptr<half, 3>, !llvm.i64) -> !llvm.ptr<half, 3>
    // CHECK-NEXT: %[[CBASE:.*]] = llvm.bitcast %[[NBASE]] : !llvm.ptr<half, 3> to !llvm.ptr<i32, 3>
    // CHECK-NEXT: %[[STRIDE:.*]] = llvm.mlir.constant(32 : i64) : !llvm.i32
    // CHECK-NEXT: %[[FRAG:.*]] = nvvm.wmma.m16n16k16.load %[[CBASE]], %[[STRIDE]] {operand = "AOp"} : !llvm.ptr<i32, 3>, !llvm.i32 -> !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    // CHECK-NEXT: %[[STOREADDRR:.*]] = llvm.getelementptr %[[STOREADDR]][%{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i64) -> !llvm.ptr<vec<16 x half>>
    // CHECK-NEXT: %[[CSTOREADDR:.*]] = llvm.bitcast %[[STOREADDRR]] : !llvm.ptr<vec<16 x half>> to !llvm.ptr<i32>
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
    %sg = alloca(){alignment = 32} : memref<32x32xf16, 3>
    %D = alloca() : memref<1xvector<8xf16>, 5>
    %i = constant 16 : i64
    %j = constant 16 : i64
    %c0 = constant 0 : i64
    gpu.subgroup_mma_store_matrix %D[%c0], %sg[%i,%j] {ldm = 32 : i64} : memref<1xvector<8xf16>, 5>, memref<32x32xf16, 3>

    // CHECK: %[[OFF:.*]] = llvm.mlir.constant(16 : i64) : !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(0 : i64) : !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(32 : index) : !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(32 : index) : !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(1024 : index) : !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.mlir.null : !llvm.ptr<half, 3>
    // CHECK-NEXT: {{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<half, 3>, !llvm.i64) -> !llvm.ptr<half, 3>
    // CHECK-NEXT: {{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<half, 3> to !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.alloca {{.*}} x !llvm.half {alignment = 32 : i64} : (!llvm.i64) -> !llvm.ptr<half, 3>
    // CHECK-NEXT: {{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.mlir.null : !llvm.ptr<vec<8 x half>, 5>
    // CHECK-NEXT: {{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<vec<8 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<8 x half>, 5>
    // CHECK-NEXT: {{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<vec<8 x half>, 5> to !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.alloca {{.*}} x !llvm.vec<8 x half> : (!llvm.i64) -> !llvm.ptr<vec<8 x half>, 5>
    // CHECK-NEXT: {{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %[[SRCADDRR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %[[BASEADDR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %[[OFFSETTT:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-NEXT: %[[LDM:.*]] = llvm.mlir.constant(32 : i64) : !llvm.i64
    // CHECK-NEXT: %[[OILDM:.*]] = llvm.mul %[[LDM]], %[[OFF]] : !llvm.i64
    // CHECK-NEXT: %[[OIJLDM:.*]] = llvm.add %[[OILDM]], %[[OFF]] : !llvm.i64
    // CHECK-NEXT: %[[TOFFSET:.*]] = llvm.add %[[OIJLDM]], %[[OFFSETTT]] : !llvm.i64
    // CHECK-NEXT: %[[LADDR:.*]] = llvm.getelementptr %[[BASEADDR]][%[[TOFFSET]]] : (!llvm.ptr<half, 3>, !llvm.i64) -> !llvm.ptr<half, 3>
    // CHECK-NEXT: %[[CADDR:.*]] = llvm.bitcast %[[LADDR]] : !llvm.ptr<half, 3> to !llvm.ptr<i32, 3>
    // CHECK-NEXT: %[[BASE:.*]] = llvm.getelementptr %[[SRCADDRR]][%1] : (!llvm.ptr<vec<8 x half>>, !llvm.i64) -> !llvm.ptr<vec<8 x half>>
    // CHECK-NEXT: %[[SRCADDR:.*]] = llvm.bitcast %[[BASE]] : !llvm.ptr<vec<8 x half>> to !llvm.ptr<i32>
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(0 : ui32) : !llvm.i32
    // CHECK-NEXT: %[[ADDR0:.*]] = llvm.getelementptr %[[SRCADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    // CHECK-NEXT: %[[EL0I32:.*]] = llvm.load %[[ADDR0]] : !llvm.ptr<i32>
    // CHECK-NEXT: %[[EL0:.*]] = llvm.bitcast %[[EL0I32]] : !llvm.i32 to !llvm.vec<2 x half>
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(1 : ui32) : !llvm.i32
    // CHECK-NEXT: %[[ADDR1:.*]] = llvm.getelementptr %[[SRCADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    // CHECK-NEXT: %[[EL1I32:.*]] = llvm.load %[[ADDR1]] : !llvm.ptr<i32>
    // CHECK-NEXT: %[[EL1:.*]] = llvm.bitcast %[[EL1I32]] : !llvm.i32 to !llvm.vec<2 x half>
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(2 : ui32) : !llvm.i32
    // CHECK-NEXT: %[[ADDR2:.*]] = llvm.getelementptr %[[SRCADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    // CHECK-NEXT: %[[EL2I32:.*]] = llvm.load %[[ADDR2]] : !llvm.ptr<i32>
    // CHECK-NEXT: %[[EL2:.*]] = llvm.bitcast %[[EL2I32]] : !llvm.i32 to !llvm.vec<2 x half>
    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(3 : ui32) : !llvm.i32
    // CHECK-NEXT: %[[ADDR3:.*]] = llvm.getelementptr %[[SRCADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    // CHECK-NEXT: %[[EL3I32:.*]] = llvm.load %[[ADDR3]] : !llvm.ptr<i32>
    // CHECK-NEXT: %[[EL3:.*]] = llvm.bitcast %[[EL3I32]] : !llvm.i32 to !llvm.vec<2 x half>
    // CHECK-NEXT: %[[STRIDE:.*]] = llvm.mlir.constant(32 : i64) : !llvm.i32
    // CHECK-NEXT: nvvm.wmma.m16n16k16.store %[[CADDR]], %[[EL0]], %[[EL1]], %[[EL2]], %[[EL3]], %[[STRIDE]] : !llvm.ptr<i32, 3>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.i32
    return
  }
}

// -----

gpu.module @test_module {

  // CHECK-LABEL: func @gpu_wmma_mma_op()
  func @gpu_wmma_mma_op() -> () {
    %A = alloca() : memref<1xvector<16xf16>, 5>
    %B = alloca() : memref<1xvector<16xf16>, 5>
    %C = alloca() : memref<1xvector<8xf16>, 5>
    %D = alloca() : memref<1xvector<8xf16>, 5>
    %c0 = constant 0 : i64

    gpu.subgroup_mma_compute %A[%c0], %B[%c0], %C[%c0], %D[%c0] : memref<1xvector<16xf16>, 5>, memref<1xvector<16xf16>, 5>, memref<1xvector<8xf16>, 5>, memref<1xvector<8xf16>, 5>

    //CHECK-NEXT:%[[c0:.*]] = llvm.mlir.constant(0 : i64) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.mlir.null : !llvm.ptr<vec<16 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<vec<16 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<16 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<vec<16 x half>, 5> to !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.alloca {{.*}} x !llvm.vec<16 x half> : (!llvm.i64) -> !llvm.ptr<vec<16 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.mlir.null : !llvm.ptr<vec<16 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<vec<16 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<16 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<vec<16 x half>, 5> to !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.alloca {{.*}} x !llvm.vec<16 x half> : (!llvm.i64) -> !llvm.ptr<vec<16 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.mlir.null : !llvm.ptr<vec<8 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<vec<8 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<8 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<vec<8 x half>, 5> to !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.alloca {{.*}} x !llvm.vec<8 x half> : (!llvm.i64) -> !llvm.ptr<vec<8 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.mlir.null : !llvm.ptr<vec<8 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<vec<8 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<8 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<vec<8 x half>, 5> to !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.alloca {{.*}} x !llvm.vec<8 x half> : (!llvm.i64) -> !llvm.ptr<vec<8 x half>, 5>
    //CHECK-NEXT:{{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:%[[AADDRR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:%[[BADDRR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:%[[CADDRR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:%[[DADDRR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<vec<8 x half>>, ptr<vec<8 x half>>, i64, array<1 x i64>, array<1 x i64>)>
    //CHECK-NEXT:%[[ABASE:.*]] = llvm.getelementptr %[[AADDRR]][%[[c0]]] : (!llvm.ptr<vec<16 x half>>, !llvm.i64) -> !llvm.ptr<vec<16 x half>>
    //CHECK-NEXT:%[[AADDR:.*]] = llvm.bitcast %[[ABASE]] : !llvm.ptr<vec<16 x half>> to !llvm.ptr<i32>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[A0I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[A0:.*]] = llvm.bitcast %[[A0I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[A1I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[A1:.*]] = llvm.bitcast %[[A1I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(2 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[A2I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[A2:.*]] = llvm.bitcast %[[A2I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(3 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[A3I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[A3:.*]] = llvm.bitcast %[[A3I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(4 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[A4I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[A4:.*]] = llvm.bitcast %[[A4I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(5 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[A5I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[A5:.*]] = llvm.bitcast %[[A5I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(6 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[A6I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[A6:.*]] = llvm.bitcast %[[A6I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(7 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[A7I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[A7:.*]] = llvm.bitcast %[[A7I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:%[[BBASE:.*]] = llvm.getelementptr %[[BADDRR]][%[[c0]]] : (!llvm.ptr<vec<16 x half>>, !llvm.i64) -> !llvm.ptr<vec<16 x half>>
    //CHECK-NEXT:%[[BADDR:.*]] = llvm.bitcast %[[BBASE]] : !llvm.ptr<vec<16 x half>> to !llvm.ptr<i32>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[B0I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[B0:.*]] = llvm.bitcast %[[B0I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[B1I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[B1:.*]] = llvm.bitcast %[[B1I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(2 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[B2I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[B2:.*]] = llvm.bitcast %[[B2I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(3 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[B3I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[B3:.*]] = llvm.bitcast %[[B3I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(4 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[B4I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[B4:.*]] = llvm.bitcast %[[B4I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(5 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[B5I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[B5:.*]] = llvm.bitcast %[[B5I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(6 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[B6I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[B6:.*]] = llvm.bitcast %[[B6I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(7 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[B7I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[B7:.*]] = llvm.bitcast %[[B7I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:%[[CBASE:.*]] = llvm.getelementptr %[[CADDRR]][%[[c0]]] : (!llvm.ptr<vec<8 x half>>, !llvm.i64) -> !llvm.ptr<vec<8 x half>>
    //CHECK-NEXT:%[[CADDR:.*]] = llvm.bitcast %[[CBASE]] : !llvm.ptr<vec<8 x half>> to !llvm.ptr<i32>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[CADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[C0I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[C0:.*]] = llvm.bitcast %[[C0I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[CADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[C1I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[C1:.*]] = llvm.bitcast %[[C1I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(2 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[CADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[C2I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[C2:.*]] = llvm.bitcast %[[C2I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(3 : ui32) : !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[CADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:%[[C3I32:.*]] = llvm.load {{.*}} : !llvm.ptr<i32>
    //CHECK-NEXT:%[[C3:.*]] = llvm.bitcast %[[C3I32]] : !llvm.i32 to !llvm.vec<2 x half>
    //CHECK-NEXT:{{.*}} = nvvm.wmma.m16n16k16.mma %[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]], %[[A6]], %[[A7]], %[[B0]], %[[B1]], %[[B2]], %[[B3]], %[[B4]], %[[B5]], %[[B6]], %[[B7]], %[[C0]], %[[C1]], %[[C2]], %[[C3]] : !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half> -> !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    //CHECK-NEXT:%[[DBASE:.*]] = llvm.getelementptr %[[DADDRR]][%[[c0]]] : (!llvm.ptr<vec<8 x half>>, !llvm.i64) -> !llvm.ptr<vec<8 x half>>
    //CHECK-NEXT:%[[DADDR:.*]] = llvm.bitcast %[[DBASE]] : !llvm.ptr<vec<8 x half>> to !llvm.ptr<i32>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[0 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    //CHECK-NEXT:%[[D0:.*]] = llvm.bitcast {{.*}} : !llvm.vec<2 x half> to !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : ui32) : !llvm.i32
    //CHECK-NEXT:%[[DADDR0:.*]] = llvm.getelementptr %[[DADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT:llvm.store %[[D0]], %[[DADDR0]] : !llvm.ptr<i32>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[1 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    //CHECK-NEXT:%[[D1:.*]] = llvm.bitcast {{.*}} : !llvm.vec<2 x half> to !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : ui32) : !llvm.i32
    //CHECK-NEXT:%[[DADDR1:.*]] = llvm.getelementptr %[[DADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT: llvm.store %[[D1]], %[[DADDR1]] : !llvm.ptr<i32>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[2 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    //CHECK-NEXT:%[[D2:.*]] = llvm.bitcast {{.*}} : !llvm.vec<2 x half> to !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(2 : ui32) : !llvm.i32
    //CHECK-NEXT:%[[DADDR2:.*]] = llvm.getelementptr %[[DADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT: llvm.store %[[D2]], %[[DADDR2]] : !llvm.ptr<i32>
    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[3 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    //CHECK-NEXT:%[[D3:.*]] = llvm.bitcast {{.*}} : !llvm.vec<2 x half> to !llvm.i32
    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(3 : ui32) : !llvm.i32
    //CHECK-NEXT:%[[DADDR3:.*]] = llvm.getelementptr %[[DADDR]][{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
    //CHECK-NEXT: llvm.store %[[D3]], %[[DADDR3]] : !llvm.ptr<i32>
    //CHECK-NEXT:{{.*}} llvm.return

    return
  }
}
