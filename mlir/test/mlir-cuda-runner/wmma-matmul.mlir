// RUN: mlir-opt -convert-scf-to-std %s | mlir-cuda-runner --sm=sm_75 --shared-libs=%cuda_wrapper_library_dir/libcuda-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// Test case to check the working of Tensor cores on Nvidia GPU's. The kernel has already
// been outlined to prevent crashing due to introduction of an empty basic block by --gpu-
// kernel-outling.
module attributes {gpu.container_module}  {
  func @main() {
    %0 = alloc() {alginment = 32} : memref<16x16xf16>
    %22 = alloc() {alginment = 32} : memref<16x16xf16>
    %1 = alloc() {alginment = 32} : memref<16x16xf32>

    %f1 = constant 1.0e+00 : f16
    %f0 = constant 0.0e+00 : f16
    %c0 = constant 0 : index
    %c16 = constant 16 : index
    %c32 = constant 32 : index
    %c1 = constant 1 : index

    // Intialize the f32 matrix with ones.
    scf.for %arg0 = %c0 to %c16 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        store %f1, %0[%arg0, %arg1] : memref<16x16xf16>
      }
    }
    // Intialize the f32 matrix with ones.
    scf.for %arg0 = %c0 to %c16 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        store %f0, %22[%arg0, %arg1] : memref<16x16xf16>
      }
    }

    %2 = memref_cast %0 : memref<16x16xf16> to memref<*xf16>
    %33 = memref_cast %22 : memref<16x16xf16> to memref<*xf16>
    %3 = memref_cast %1 : memref<16x16xf32> to memref<*xf32>
    gpu.host_register %2 : memref<*xf16>
    gpu.host_register %33 : memref<*xf16>

    // Print memref to see if initialize with one.
    //call @print_memref_f32(%3) : (memref<*xf32>) -> ()

    scf.for %arg0 = %c0 to %c16 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %6 = load %0[%arg0, %arg1] : memref<16x16xf16>
        %7 = fpext %6 : f16 to f32
        store %7, %1[%arg0, %arg1] : memref<16x16xf32>
      }
    }
    // Print just before the computation to see if %0 was intitialized
    // properly.
    //call @print_memref_f32(%3) : (memref<*xf32>) -> ()

    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1) args(%0 : memref<16x16xf16>, %22 : memref<16x16xf16>)

    scf.for %arg0 = %c0 to %c16 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %6 = load %0[%arg0, %arg1] : memref<16x16xf16>
        %7 = fpext %6 : f16 to f32
        store %7, %1[%arg0, %arg1] : memref<16x16xf32>
      }
    }

    // Print the memref after computation.
    call @print_memref_f32(%3) : (memref<*xf32>) -> ()
    // CHECK: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16]
    return
  }

  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<16x16xf16>, %arg22 : memref<16x16xf16>) workgroup(%arg3 : memref<8xvector<2xf16>, 3>) private(%arg4 : memref<1xvector<16xf16>, 5>, %arg5 : memref<1xvector<16xf16>, 5>, %arg6 : memref<1xvector<8xf16>, 5>, %arg7 : memref<1xvector<8xf16>, 5>, %arg8 : memref<2x6xi32, 5>) kernel {
      %c0_i64 = constant 0 : i64
      %c0 = constant 0 : index
      %c1 = constant 1.00e+00 : f16
      gpu.subgroup_mma_load_matrix %arg0[%c0_i64, %c0_i64], %arg4[%c0_i64] {ldm = 16 : i64, operand = "AOp"} : memref<16x16xf16>, memref<1xvector<16xf16>, 5>
      gpu.subgroup_mma_load_matrix %arg0[%c0_i64, %c0_i64], %arg5[%c0_i64] {ldm = 16 : i64, operand = "BOp"} : memref<16x16xf16>, memref<1xvector<16xf16>, 5>
      gpu.subgroup_mma_load_matrix %arg22[%c0_i64, %c0_i64], %arg6[%c0_i64] {ldm = 16 : i64, operand = "COp"} : memref<16x16xf16>, memref<1xvector<8xf16>, 5>

      gpu.subgroup_mma_compute %arg4[%c0_i64], %arg5[%c0_i64], %arg6[%c0_i64], %arg7[%c0_i64] : memref<1xvector<16xf16>, 5>, memref<1xvector<16xf16>, 5>, memref<1xvector<8xf16>, 5>, memref<1xvector<8xf16>, 5>

      gpu.subgroup_mma_store_matrix %arg7[%c0_i64], %arg0[%c0_i64, %c0_i64] {ldm = 16 : i64} : memref<1xvector<8xf16>, 5>, memref<16x16xf16>
      gpu.return
    }
  }

  func private @print_memref_f32(memref<*xf32>)
}
