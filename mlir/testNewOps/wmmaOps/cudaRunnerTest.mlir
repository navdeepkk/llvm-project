// RUN: mlir-opt --gpu-kernel-outlining --convert-gpu-to-nvvm %s | FileCheck %s

func @main() {
  %l = alloc() : memref<32x32xf16, 3>
  %c1 = constant 1 : index
  %c32 = constant 32 : index

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c32, %block_y = %c1, %block_z = %c1) {
  %A = alloca() {alignment = 32} : memref<8xvector<2xf16>, 5>
  gpu.subgroup_mma_load_matrix %l, %A {layout = "RowMajor", srcMemSpace = 3 : ui8, operand = "AOp", srcOffsetJ = 16 : index, srcOffsetI = 16 : index, stride = 16 : ui16, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : memref<32x32xf16, 3>, memref<8xvector<2xf16>, 5>
  
  gpu.terminator
  }

  return
}
