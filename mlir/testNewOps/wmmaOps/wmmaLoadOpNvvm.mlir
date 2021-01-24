module {
  llvm.func @gpu_wmma_ops(% arg0 : !llvm.ptr<i32, 3>, % arg1 : !llvm.i32) {
    %0 = nvvm.wmma.load %arg0, %arg1 {ldm = 32 : i64, operand = "AOp", wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : !llvm.ptr<i32, 3>, !llvm.i32 -> !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    
    llvm.return
  }
}
