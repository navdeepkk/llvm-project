func @matmul(){
  %m1 = alloc() : memref<1024x1024xf16>
  %m2 = alloc() : memref<1024x1024xf16>
  %m3 = alloc() : memref<1024x1024xf16>
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %m = dim %m1, %c0: memref<1024x1024xf16> 
  %n = dim %m2, %c1: memref<1024x1024xf16> 
  %k = dim %m1, %c1: memref<1024x1024xf16>
  %A = alloca() : memref<8xvector<2xf16>, 5>
  %B = alloca() : memref<8xvector<2xf16>, 5>
  %C = alloca() : memref<4xvector<2xf16>, 5>
  affine.parallel(%i, %j) = (0, 0) to (%m, %n) step(16, 16) {
    gpu.subgroup_mma_load_matrix %m1, %A {operand = "AOp", srcOffsetJ = 16 : i64, srcOffsetI = 16 : i64, ldm = 32 : i64, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : memref<1024x1024xf16>, memref<8xvector<2xf16>, 5>
    gpu.subgroup_mma_load_matrix %m2, %B {operand = "BOp", srcOffsetJ = 16 : i64, srcOffsetI = 16 : i64, ldm = 32 : i64, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : memref<1024x1024xf16>, memref<8xvector<2xf16>, 5>
    gpu.subgroup_mma_load_matrix %m3, %C {operand = "COp", srcOffsetJ = 16 : i64, srcOffsetI = 16 : i64, ldm = 32 : i64, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : memref<1024x1024xf16>, memref<4xvector<2xf16>, 5>
    gpu.subgroup_mma_compute %A, %B, %C, %C {ldm = 32 : ui16, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : memref<8xvector<2xf16>, 5>, memref<8xvector<2xf16>, 5>, memref<4xvector<2xf16>, 5>, memref<4xvector<2xf16>, 5>
    gpu.subgroup_mma_store_matrix %C, %m3 {dstOffsetJ = 16 : i64, dstOffsetI = 16 : i64, ldm = 32 : i64, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : memref<4xvector<2xf16>, 5>, memref<1024x1024xf16>
  }
  return
}
