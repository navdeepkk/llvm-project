// RUN: mlir-opt --gpu-kernel-outlining --convert-gpu-to-nvvm %s | FileCheck %s

func @main() {
  %l = alloc() : memref<32x32xf16, 3>
  %s = alloc() : memref<32x32xf16, 3>
  %c1 = constant 1 : index

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1, %block_z = %c1) {
  %A = alloca() {alignment = 32} : memref<8xvector<2xf16>>
  %B = alloca() {alignment = 32} : memref<8xvector<2xf16>>
  %C = alloca() {alignment = 32} : memref<4xvector<2xf16>>
  %D = alloca() {alignment = 32} : memref<4xvector<2xf16>>
  gpu.subgroup_mma_load_matrix %l, %A {layout = "RowMajor", srcMemSpace = 3 : ui8, operand = "AOp", srcOffsetJ = 16 : index, srcOffsetI = 16 : index, stride = 16 : ui16, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : memref<32x32xf16, 3>, memref<8xvector<2xf16>>
  //gpu.subgroup_mma_load_matrix %l, %B {layout = "RowMajor", srcMemSpace = 3 : ui8, operand = "BOp", srcOffsetJ = 16 : index, srcOffsetI = 16 : index, stride = 16 : ui16, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : memref<32x32xf16>, memref<16xvector<2xf16>>
  //gpu.subgroup_mma_load_matrix %l, %C {layout = "RowMajor", srcMemSpace = 3 : ui8, operand = "COp", srcOffsetJ = 16 : index, srcOffsetI = 16 : index, stride = 16 : ui16, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : memref<32x32xf16>, memref<16xvector<2xf16>>
  
  gpu.subgroup_mma_compute %A, %B, %C, %D {stride = 16 : ui16, BLayout = "RowMajor",ALayout = "RowMajor", wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : memref<8xvector<2xf16>>, memref<8xvector<2xf16>>, memref<4xvector<2xf16>>, memref<4xvector<2xf16>>
  
  gpu.subgroup_mma_store_matrix %D, %s {layout = "RowMajor", dstMemSpace = 3 : ui8, dstOffsetJ = 16 : index, dstOffsetI = 16 : index, stride = 16 : ui16, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : memref<4xvector<2xf16>>, memref<32x32xf16, 3>
    
  gpu.terminator
  }

  return
}
