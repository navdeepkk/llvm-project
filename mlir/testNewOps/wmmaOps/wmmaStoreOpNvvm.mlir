module {
  llvm.func @gpu_wmma_ops(% arg0
                          : !llvm.ptr<i32, 3>, % arg1
                          : !llvm.vec<2 x half>, % arg2
                          : !llvm.vec<2 x half>, % arg3
                          : !llvm.vec<2 x half>, % arg4
                          : !llvm.vec<2 xhalf>, % arg5
                          : !llvm.i32) {
    nvvm.wmma.store % arg0, % arg1, % arg2, % arg3, % arg4,
        % arg5{ldm = 32 : i64, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16}
        : !llvm.ptr<i32, 3>,
        !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>,
        !llvm.vec<2 x half>, !llvm.i32 llvm.return
  }
}
