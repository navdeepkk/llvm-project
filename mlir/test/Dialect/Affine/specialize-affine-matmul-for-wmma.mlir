// RUN: mlir-opt --test-specialize-affine-matmul-for-wmma="accum=f32 load-store-width=128" --canonicalize --cse %s | FileCheck %s

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 16)>
#map2 = affine_map<(d0) -> (d0 + 128)>
module  {
  global_memref "public" @frag_A : memref<128x16xf16, 3>
  global_memref "public" @frag_B : memref<16x128xf16, 3>
  func @main() {
    %cst = constant 1.600000e+01 : f16
    %cst_0 = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2147483648_i64 = constant 2147483648 : i64
    %c1024 = constant 1024 : index
    %0 = alloc() : memref<1024x1024xf16>
    %1 = alloc() : memref<1024x1024xf16>
    %2 = alloc() : memref<1024x1024xf32>
    affine.for %arg0 = 0 to 1024 step 128 {
      affine.for %arg1 = 0 to 1024 step 128 {
        %14 = get_global_memref @frag_B : memref<16x128xf16, 3>
        %15 = get_global_memref @frag_A : memref<128x16xf16, 3>
        affine.for %arg2 = 0 to 1024 step 16 {
          affine.for %arg3 = #map0(%arg2) to #map1(%arg2) {
            affine.for %arg4 = #map0(%arg1) to #map2(%arg1) {
              %16 = affine.load %1[%arg3, %arg4] : memref<1024x1024xf16>
              affine.store %16, %14[%arg3 - %arg2, %arg4 - %arg1] : memref<16x128xf16, 3>
            }
          } {isCopyLoopNest = true}
          affine.for %arg3 = #map0(%arg0) to #map2(%arg0) {
            affine.for %arg4 = #map0(%arg2) to #map1(%arg2) {
              %16 = affine.load %0[%arg3, %arg4] : memref<1024x1024xf16>
              affine.store %16, %15[%arg3 - %arg0, %arg4 - %arg2] : memref<128x16xf16, 3>
            }
          } {isCopyLoopNest = true}
          affine.for %arg3 = 0 to 128 step 32 {
            affine.for %arg4 = 0 to 128 step 32 {
              affine.for %arg5 = 0 to 16 step 16 {
                affine.for %arg6 = 0 to 32 {
                  affine.for %arg7 = 0 to 32 {
                    affine.for %arg8 = 0 to 16 {
                      %16 = affine.load %15[%arg3 + %arg6, %arg5 + %arg8] : memref<128x16xf16, 3>
                      %17 = affine.load %14[%arg5 + %arg8, %arg4 + %arg7] : memref<16x128xf16, 3>
                      %18 = affine.load %2[%arg0 + %arg3 + %arg6, %arg1 + %arg4 + %arg7] : memref<1024x1024xf32>
                      %19 = mulf %16, %17 : f16
                      %20 = fpext %19 : f16 to f32
                      %21 = addf %18, %20 : f32
                      affine.store %21, %2[%arg0 + %arg3 + %arg6, %arg1 + %arg4 + %arg7] : memref<1024x1024xf32>
                    }
                  }
                }
              }
            }
          } {isComputeLoopNest = true}
        }
      }
    }
    return
  }
}

// CHECK-DAG: #map0 = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-DAG: #map1 = affine_map<(d0, d1) -> (d0 + d1 + 16)>
// CHECK-DAG: #map2 = affine_map<(d0) -> (d0 + 16)>
// CHECK: module  {
// CHECK-NEXT:   global_memref "public" @frag_A : memref<128x16xf16, 3>
// CHECK-NEXT:   global_memref "public" @frag_B : memref<16x128xf16, 3>
// CHECK-NEXT:   func @main() {
// CHECK-NEXT:     %0 = alloc() {alignment = 16 : i64} : memref<1024x1024xf16>
// CHECK-NEXT:     %1 = memref_vector_cast %0 : memref<1024x1024xf16> to memref<1024x128xvector<8xf16>>
// CHECK-NEXT:     %2 = alloc() {alignment = 16 : i64} : memref<1024x1024xf16>
// CHECK-NEXT:     %3 = memref_vector_cast %2 : memref<1024x1024xf16> to memref<1024x128xvector<8xf16>>
// CHECK-NEXT:     %4 = alloc() : memref<1024x1024xf32>
// CHECK-NEXT:     affine.parallel (%arg0) = (0) to (1024) step (128) {
// CHECK-NEXT:       affine.parallel (%arg1) = (0) to (1024) step (128) {
// CHECK-NEXT:         %5 = get_global_memref @frag_B : memref<16x128xf16, 3>
// CHECK-NEXT:         %6 = memref_vector_cast %5 : memref<16x128xf16, 3> to memref<16x16xvector<8xf16>, 3>
// CHECK-NEXT:         %7 = get_global_memref @frag_A : memref<128x16xf16, 3>
// CHECK-NEXT:         %8 = memref_vector_cast %7 : memref<128x16xf16, 3> to memref<128x2xvector<8xf16>, 3>
// CHECK-NEXT:         affine.parallel (%arg2) = (0) to (128) step (32) {
// CHECK-NEXT:           affine.parallel (%arg3) = (0) to (128) step (32) {
// CHECK-NEXT:             %9 = affine.apply #map0(%arg0, %arg2)
// CHECK-NEXT:             %10 = affine.apply #map0(%arg1, %arg3)
// CHECK-NEXT:             %11 = gpu.subgroup_mma_load_matrix %4[%9, %10] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf32> -> !gpu.mmafragment<8, f32>
// CHECK-NEXT:             %12 = affine.apply #map1(%arg0, %arg2)
// CHECK-NEXT:             %13 = gpu.subgroup_mma_load_matrix %4[%12, %10] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf32> -> !gpu.mmafragment<8, f32>
// CHECK-NEXT:             %14 = affine.apply #map1(%arg1, %arg3)
// CHECK-NEXT:             %15 = gpu.subgroup_mma_load_matrix %4[%9, %14] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf32> -> !gpu.mmafragment<8, f32>
// CHECK-NEXT:             %16 = gpu.subgroup_mma_load_matrix %4[%12, %14] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf32> -> !gpu.mmafragment<8, f32>
// CHECK-NEXT:             %17:4 = affine.for %arg4 = 0 to 1024 step 16 iter_args(%arg5 = %11, %arg6 = %13, %arg7 = %15, %arg8 = %16) -> (!gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>) {
// CHECK-NEXT:               gpu.barrier
// CHECK-NEXT:               affine.parallel (%arg9) = (%arg4) to (%arg4 + 16) {
// CHECK-NEXT:                 affine.parallel (%arg10) = (%arg1) to (%arg1 + 128) step (8) {
// CHECK-NEXT:                   %19 = affine.load %3[%arg9, %arg10 floordiv 8] : memref<1024x128xvector<8xf16>>
// CHECK-NEXT:                   affine.store %19, %6[%arg9 - %arg4, (%arg10 - %arg1) floordiv 8] : memref<16x16xvector<8xf16>, 3>
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:               gpu.barrier
// CHECK-NEXT:               affine.parallel (%arg9) = (%arg0) to (%arg0 + 128) {
// CHECK-NEXT:                 affine.parallel (%arg10) = (%arg4) to (%arg4 + 16) step (8) {
// CHECK-NEXT:                   %19 = affine.load %1[%arg9, %arg10 floordiv 8] : memref<1024x128xvector<8xf16>>
// CHECK-NEXT:                   affine.store %19, %8[%arg9 - %arg0, (%arg10 - %arg4) floordiv 8] : memref<128x2xvector<8xf16>, 3>
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:               gpu.barrier
// CHECK-NEXT:               %18:4 = affine.for %arg9 = 0 to 16 step 16 iter_args(%arg10 = %arg5, %arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8) -> (!gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>) {
// CHECK-NEXT:                 %19 = gpu.subgroup_mma_load_matrix %7[%arg2, %arg9] {leadDimension = 16 : index, operand = "AOp"} : memref<128x16xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
// CHECK-NEXT:                 %20 = gpu.subgroup_mma_load_matrix %5[%arg9, %arg3] {leadDimension = 128 : index, operand = "BOp"} : memref<16x128xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
// CHECK-NEXT:                 %21 = gpu.subgroup_mma_compute %19, %20, %arg10 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
// CHECK-NEXT:                 %22 = affine.apply #map2(%arg2)
// CHECK-NEXT:                 %23 = gpu.subgroup_mma_load_matrix %7[%22, %arg9] {leadDimension = 16 : index, operand = "AOp"} : memref<128x16xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
// CHECK-NEXT:                 %24 = gpu.subgroup_mma_load_matrix %5[%arg9, %arg3] {leadDimension = 128 : index, operand = "BOp"} : memref<16x128xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
// CHECK-NEXT:                 %25 = gpu.subgroup_mma_compute %23, %24, %arg11 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
// CHECK-NEXT:                 %26 = gpu.subgroup_mma_load_matrix %7[%arg2, %arg9] {leadDimension = 16 : index, operand = "AOp"} : memref<128x16xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
// CHECK-NEXT:                 %27 = affine.apply #map2(%arg3)
// CHECK-NEXT:                 %28 = gpu.subgroup_mma_load_matrix %5[%arg9, %27] {leadDimension = 128 : index, operand = "BOp"} : memref<16x128xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
// CHECK-NEXT:                 %29 = gpu.subgroup_mma_compute %26, %28, %arg12 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
// CHECK-NEXT:                 %30 = gpu.subgroup_mma_load_matrix %7[%22, %arg9] {leadDimension = 16 : index, operand = "AOp"} : memref<128x16xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
// CHECK-NEXT:                 %31 = gpu.subgroup_mma_load_matrix %5[%arg9, %27] {leadDimension = 128 : index, operand = "BOp"} : memref<16x128xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
// CHECK-NEXT:                 %32 = gpu.subgroup_mma_compute %30, %31, %arg13 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
// CHECK-NEXT:                 affine.yield %21, %25, %29, %32 : !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>
// CHECK-NEXT:               }
// CHECK-NEXT:               affine.yield %18#0, %18#1, %18#2, %18#3 : !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>
// CHECK-NEXT:             }
// CHECK-NEXT:             gpu.subgroup_mma_store_matrix %17#0, %4[%9, %10] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf32>
// CHECK-NEXT:             gpu.subgroup_mma_store_matrix %17#1, %4[%12, %10] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf32>
// CHECK-NEXT:             gpu.subgroup_mma_store_matrix %17#2, %4[%9, %14] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf32>
// CHECK-NEXT:             gpu.subgroup_mma_store_matrix %17#3, %4[%12, %14] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }
