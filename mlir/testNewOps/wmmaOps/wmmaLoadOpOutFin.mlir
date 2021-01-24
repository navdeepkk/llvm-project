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
      %9 = llvm.alloca %8 x !llvm.half : (!llvm.i64) -> !llvm.ptr<half, 3>
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
      %21 = llvm.mlir.null : !llvm.ptr<vec<16 x half>, 5>
      %22 = llvm.getelementptr %21[%19] : (!llvm.ptr<vec<16 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<16 x half>, 5>
      %23 = llvm.ptrtoint %22 : !llvm.ptr<vec<16 x half>, 5> to !llvm.i64
      %24 = llvm.alloca %23 x !llvm.vec<16 x half> : (!llvm.i64) -> !llvm.ptr<vec<16 x half>, 5>
      %25 = llvm.mlir.undef : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %26 = llvm.insertvalue %24, %25[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %27 = llvm.insertvalue %24, %26[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %28 = llvm.mlir.constant(0 : index) : !llvm.i64
      %29 = llvm.insertvalue %28, %27[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %30 = llvm.insertvalue %19, %29[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %31 = llvm.insertvalue %20, %30[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %32 = llvm.extractvalue %18[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %33 = llvm.extractvalue %18[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %34 = llvm.extractvalue %18[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %35 = llvm.extractvalue %18[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %36 = llvm.extractvalue %18[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %37 = llvm.extractvalue %18[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %38 = llvm.extractvalue %18[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
      %39 = llvm.extractvalue %31[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %40 = llvm.extractvalue %31[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %41 = llvm.extractvalue %31[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %42 = llvm.extractvalue %31[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %43 = llvm.extractvalue %31[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
      %44 = llvm.mlir.constant(32 : i64) : !llvm.i64
      %45 = llvm.mul %44, %0 : !llvm.i64
      %46 = llvm.add %45, %0 : !llvm.i64
      %47 = llvm.add %46, %34 : !llvm.i64
      %48 = llvm.getelementptr %33[%47] : (!llvm.ptr<half, 3>, !llvm.i64) -> !llvm.ptr<half, 3>
      %49 = llvm.bitcast %48 : !llvm.ptr<half, 3> to !llvm.ptr<i32, 3>
      %50 = llvm.mlir.constant(32 : i64) : !llvm.i32
      %51 = nvvm.wmma.m16n16k16.load %49, %50 {operand = "AOp"} : !llvm.ptr<i32, 3>, !llvm.i32 -> !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %52 = llvm.getelementptr %40[%1] : (!llvm.ptr<vec<16 x half>>, !llvm.i64) -> !llvm.ptr<vec<16 x half>>
      %53 = llvm.bitcast %52 : !llvm.ptr<vec<16 x half>> to !llvm.ptr<i32>
      %54 = llvm.extractvalue %51[0 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %55 = llvm.bitcast %54 : !llvm.vec<2 x half> to !llvm.i32
      %56 = llvm.mlir.constant(0 : ui32) : !llvm.i32
      %57 = llvm.getelementptr %53[%56] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %55, %57 : !llvm.ptr<i32>
      %58 = llvm.extractvalue %51[1 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %59 = llvm.bitcast %58 : !llvm.vec<2 x half> to !llvm.i32
      %60 = llvm.mlir.constant(1 : ui32) : !llvm.i32
      %61 = llvm.getelementptr %53[%60] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %59, %61 : !llvm.ptr<i32>
      %62 = llvm.extractvalue %51[2 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %63 = llvm.bitcast %62 : !llvm.vec<2 x half> to !llvm.i32
      %64 = llvm.mlir.constant(2 : ui32) : !llvm.i32
      %65 = llvm.getelementptr %53[%64] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %63, %65 : !llvm.ptr<i32>
      %66 = llvm.extractvalue %51[3 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %67 = llvm.bitcast %66 : !llvm.vec<2 x half> to !llvm.i32
      %68 = llvm.mlir.constant(3 : ui32) : !llvm.i32
      %69 = llvm.getelementptr %53[%68] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %67, %69 : !llvm.ptr<i32>
      %70 = llvm.extractvalue %51[4 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %71 = llvm.bitcast %70 : !llvm.vec<2 x half> to !llvm.i32
      %72 = llvm.mlir.constant(4 : ui32) : !llvm.i32
      %73 = llvm.getelementptr %53[%72] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %71, %73 : !llvm.ptr<i32>
      %74 = llvm.extractvalue %51[5 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %75 = llvm.bitcast %74 : !llvm.vec<2 x half> to !llvm.i32
      %76 = llvm.mlir.constant(5 : ui32) : !llvm.i32
      %77 = llvm.getelementptr %53[%76] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %75, %77 : !llvm.ptr<i32>
      %78 = llvm.extractvalue %51[6 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %79 = llvm.bitcast %78 : !llvm.vec<2 x half> to !llvm.i32
      %80 = llvm.mlir.constant(6 : ui32) : !llvm.i32
      %81 = llvm.getelementptr %53[%80] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %79, %81 : !llvm.ptr<i32>
      %82 = llvm.extractvalue %51[7 : index] : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
      %83 = llvm.bitcast %82 : !llvm.vec<2 x half> to !llvm.i32
      %84 = llvm.mlir.constant(7 : ui32) : !llvm.i32
      %85 = llvm.getelementptr %53[%84] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
      llvm.store %83, %85 : !llvm.ptr<i32>
      llvm.return
    }
    llvm.func @_mlir_ciface_gpu_wmma_ops() {
      llvm.call @gpu_wmma_ops() : () -> ()
      llvm.return
    }
  }
}

// -----

//gpu.module @test_module {
//
//  // CHECK-LABEL: func @gpu_wmma_store_op()
//  func @gpu_wmma_store_op() -> () {
//    %sg = alloca(){alignment = 32} : memref<32x32xf16, 3>
//    %D = alloca() : memref<1xvector<8xf16>, 5>
//    %i = constant 16 : i64
//    %j = constant 16 : i64
//    %c0 = constant 0 : i64
//    gpu.subgroup_mma_store_matrix %D[%c0], %sg[%i,%j] {ldm = 32 : i64} : memref<1xvector<8xf16>, 5>, memref<32x32xf16, 3>
//
//    // CHECK: {{.*}} = llvm.mlir.constant(32 : index) : !llvm.i64
//    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(32 : index) : !llvm.i64
//    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
//    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(1024 : index) : !llvm.i64
//    // CHECK-NEXT: {{.*}} = llvm.mlir.null : !llvm.ptr<half, 3>
//    // CHECK-NEXT: {{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<half, 3>, !llvm.i64) -> !llvm.ptr<half, 3>
//    // CHECK-NEXT: {{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<half, 3> to !llvm.i64
//    // CHECK-NEXT: {{.*}} = llvm.alloca {{.*}} x !llvm.half : (!llvm.i64) -> !llvm.ptr<half, 3>
//    // CHECK-NEXT: {{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
//    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(4 : index) : !llvm.i64
//    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
//    // CHECK-NEXT: {{.*}} = llvm.mlir.null : !llvm.ptr<vec<16 x half>, 5>
//    // CHECK-NEXT: {{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<vec<16 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<16 x half>, 5>
//    // CHECK-NEXT: {{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<vec<16 x half>, 5> to !llvm.i64
//    // CHECK-NEXT: {{.*}} = llvm.alloca {{.*}} x !llvm.vec<16 x half> : (!llvm.i64) -> !llvm.ptr<vec<16 x half>, 5>
//    // CHECK-NEXT: {{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
//    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    // CHECK-NEXT: %[[SRCADDR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
//    // CHECK-NEXT: %[[BASEADDR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
//    // CHECK-NEXT: %[[OFFSETTT:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[3, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
//    // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[4, 1] : !llvm.struct<(ptr<half, 3>, ptr<half, 3>, i64, array<2 x i64>, array<2 x i64>)>
//    // CHECK-NEXT: %[[OFFI:.*]] = llvm.mlir.constant(16 : i64) : !llvm.i64
//    // CHECK-NEXT: %[[OFFJ:.*]] = llvm.mlir.constant(16 : i64) : !llvm.i64
//    // CHECK-NEXT: %[[LDM:.*]] = llvm.mlir.constant(32 : i64) : !llvm.i64
//    // CHECK-NEXT: %[[OILDM:.*]] = llvm.mul %[[LDM]], %[[OFFI]] : !llvm.i64
//    // CHECK-NEXT: %[[OIJLDM:.*]] = llvm.add %[[OILDM]], %[[OFFJ]] : !llvm.i64
//    // CHECK-NEXT: %[[TOFFSET:.*]] = llvm.add %[[OIJLDM]], %[[OFFSETTT]] : !llvm.i64
//    // CHECK-NEXT: %[[LADDR:.*]] = llvm.getelementptr %[[BASEADDR]][%[[TOFFSET]]] : (!llvm.ptr<half, 3>, !llvm.i64) -> !llvm.ptr<half, 3>
//    // CHECK-NEXT: %[[CADDR:.*]] = llvm.bitcast %[[LADDR]] : !llvm.ptr<half, 3> to !llvm.ptr<i32, 3>
//    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(0 : ui32) : !llvm.i32
//    // CHECK-NEXT: %[[ADDR0:.*]] = llvm.getelementptr %[[SRCADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    // CHECK-NEXT: %[[EL0:.*]] = llvm.load %[[ADDR0]] : !llvm.ptr<vec<16 x half>>
//    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(1 : ui32) : !llvm.i32
//    // CHECK-NEXT: %[[ADDR1:.*]] = llvm.getelementptr %[[SRCADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    // CHECK-NEXT: %[[EL1:.*]] = llvm.load %[[ADDR1]] : !llvm.ptr<vec<16 x half>>
//    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(2 : ui32) : !llvm.i32
//    // CHECK-NEXT: %[[ADDR2:.*]] = llvm.getelementptr %[[SRCADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    // CHECK-NEXT: %[[EL2:.*]] = llvm.load %[[ADDR2]] : !llvm.ptr<vec<16 x half>>
//    // CHECK-NEXT: {{.*}} = llvm.mlir.constant(3 : ui32) : !llvm.i32
//    // CHECK-NEXT: %[[ADDR3:.*]] = llvm.getelementptr %[[SRCADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    // CHECK-NEXT: %[[EL3:.*]] = llvm.load %[[ADDR3]] : !llvm.ptr<vec<16 x half>>
//    // CHECK-NEXT: %[[STRIDE:.*]] = llvm.mlir.constant(32 : i64) : !llvm.i32
//    // CHECK-NEXT: nvvm.wmma.store %[[CADDR]], %[[EL0]], %[[EL1]], %[[EL2]], %[[EL3]], %[[STRIDE]] {ldm = 32 : i64, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : !llvm.ptr<i32, 3>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.i32
//    return
//  }
//}
//
//// -----
//
//gpu.module @test_module {
//
//  // CHECK-LABEL: func @gpu_wmma_mma_op()
//  func @gpu_wmma_mma_op() -> () {
//    %A = alloca() : memref<1xvector<16xf16>, 5>
//    %B = alloca() : memref<1xvector<16xf16>, 5>
//    %C = alloca() : memref<1xvector<8xf16>, 5>
//    %D = alloca() : memref<1xvector<8xf16>, 5>
//    %c0 = constant 0 : i64
//
//    gpu.subgroup_mma_compute %A[%c0], %B[%c0], %C[%c0], %D[%c0] : memref<1xvector<16xf16>, 5>, memref<1xvector<16xf16>, 5>, memref<1xvector<8xf16>, 5>, memref<1xvector<8xf16>, 5>
//
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(8 : index) : !llvm.i64
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
//    //CHECK-NEXT:{{.*}} = llvm.mlir.null : !llvm.ptr<vec<16 x half>, 5>
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<vec<16 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<16 x half>, 5>
//    //CHECK-NEXT:{{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<vec<16 x half>, 5> to !llvm.i64
//    //CHECK-NEXT:{{.*}} = llvm.alloca {{.*}} x !llvm.vec<16 x half> : (!llvm.i64) -> !llvm.ptr<vec<16 x half>, 5>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(8 : index) : !llvm.i64
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
//    //CHECK-NEXT:{{.*}} = llvm.mlir.null : !llvm.ptr<vec<16 x half>, 5>
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<vec<16 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<16 x half>, 5>
//    //CHECK-NEXT:{{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<vec<16 x half>, 5> to !llvm.i64
//    //CHECK-NEXT:{{.*}} = llvm.alloca {{.*}} x !llvm.vec<16 x half> : (!llvm.i64) -> !llvm.ptr<vec<16 x half>, 5>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(4 : index) : !llvm.i64
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
//    //CHECK-NEXT:{{.*}} = llvm.mlir.null : !llvm.ptr<vec<16 x half>, 5>
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<vec<16 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<16 x half>, 5>
//    //CHECK-NEXT:{{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<vec<16 x half>, 5> to !llvm.i64
//    //CHECK-NEXT:{{.*}} = llvm.alloca {{.*}} x !llvm.vec<16 x half> : (!llvm.i64) -> !llvm.ptr<vec<16 x half>, 5>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(4 : index) : !llvm.i64
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : index) : !llvm.i64
//    //CHECK-NEXT:{{.*}} = llvm.mlir.null : !llvm.ptr<vec<16 x half>, 5>
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<vec<16 x half>, 5>, !llvm.i64) -> !llvm.ptr<vec<16 x half>, 5>
//    //CHECK-NEXT:{{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr<vec<16 x half>, 5> to !llvm.i64
//    //CHECK-NEXT:{{.*}} = llvm.alloca {{.*}} x !llvm.vec<16 x half> : (!llvm.i64) -> !llvm.ptr<vec<16 x half>, 5>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.undef : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : index) : !llvm.i64
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.insertvalue {{.*}}, {{.*}}[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:%[[AADDR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:%[[BADDR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:%[[CADDR:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<1 x i64>, array<1 x i64>)>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[A0:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[A1:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(2 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[A2:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(3 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[A3:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(4 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[A4:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(5 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[A5:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(6 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[A6:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(7 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[AADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[A7:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[B0:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[B1:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(2 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[B2:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(3 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[B3:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(4 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[B4:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(5 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[B5:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(6 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[B6:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(7 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[BADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[B7:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[CADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[C0:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[CADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[C1:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(2 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[CADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[C2:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(3 : ui32) : !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.getelementptr %[[CADDR]][{{.*}}] : (!llvm.ptr<vec<16 x half>>, !llvm.i32) -> !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:%[[C3:.*]] = llvm.load {{.*}} : !llvm.ptr<vec<16 x half>>
//    //CHECK-NEXT:{{.*}} = nvvm.wmma.mma %[[A0]], %[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]], %[[A6]], %[[A7]], %[[B0]], %[[B1]], %[[B2]], %[[B3]], %[[B4]], %[[B5]], %[[B6]], %[[B7]], %[[C0]], %[[C1]], %[[C2]], %[[C3]] {wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half>, !llvm.vec<16 x half> -> !llvm.struct<(vec<16 x half>, vec<16 x half>, vec<16 x half>, vec<16 x half>)>
//    //CHECK-NEXT:{{.*}} = llvm.bitcast {{.*}} : !llvm.ptr<vec<16 x half>> to !llvm.ptr<i32>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[0 : index] : !llvm.struct<(vec<16 x half>, vec<16 x half>, vec<16 x half>, vec<16 x half>)>
//    //CHECK-NEXT:%[[D0:.*]] = llvm.bitcast {{.*}} : !llvm.vec<16 x half> to !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(0 : ui32) : !llvm.i32
//    //CHECK-NEXT:%[[DADDR0:.*]] = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
//    //CHECK-NEXT:llvm.store %[[D0]], %[[DADDR0]] : !llvm.ptr<i32>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[1 : index] : !llvm.struct<(vec<16 x half>, vec<16 x half>, vec<16 x half>, vec<16 x half>)>
//    //CHECK-NEXT:%[[D1:.*]] = llvm.bitcast {{.*}} : !llvm.vec<16 x half> to !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(1 : ui32) : !llvm.i32
//    //CHECK-NEXT:%[[DADDR1:.*]] = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
//    //CHECK-NEXT: llvm.store %[[D1]], %[[DADDR1]] : !llvm.ptr<i32>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[2 : index] : !llvm.struct<(vec<16 x half>, vec<16 x half>, vec<16 x half>, vec<16 x half>)>
//    //CHECK-NEXT:%[[D2:.*]] = llvm.bitcast {{.*}} : !llvm.vec<16 x half> to !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(2 : ui32) : !llvm.i32
//    //CHECK-NEXT:%[[DADDR2:.*]] = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
//    //CHECK-NEXT: llvm.store %[[D2]], %[[DADDR2]] : !llvm.ptr<i32>
//    //CHECK-NEXT:{{.*}} = llvm.extractvalue {{.*}}[3 : index] : !llvm.struct<(vec<16 x half>, vec<16 x half>, vec<16 x half>, vec<16 x half>)>
//    //CHECK-NEXT:%[[D3:.*]] = llvm.bitcast {{.*}} : !llvm.vec<16 x half> to !llvm.i32
//    //CHECK-NEXT:{{.*}} = llvm.mlir.constant(3 : ui32) : !llvm.i32
//    //CHECK-NEXT:%[[DADDR3:.*]] = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
//    //CHECK-NEXT: llvm.store %[[D3]], %[[DADDR3]] : !llvm.ptr<i32>
//    //CHECK-NEXT:{{.*}} llvm.return
//
//    return
//  }
//}
