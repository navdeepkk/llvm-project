module {
  llvm.func @gpu_wmma_ops(% arg0
                          : !llvm.vec<2 x half>, % arg1
                          : !llvm.vec<2 x half>, % arg2
                          : !llvm.vec<2 x half>, % arg3
                          : !llvm.vec<2 x half>, % arg4
                          : !llvm.vec<2 x half>, % arg5
                          : !llvm.vec<2 x half>, % arg6
                          : !llvm.vec<2 x half>, % arg7
                          : !llvm.vec<2 x half>, % arg8
                          : !llvm.vec<2 x half>, % arg9
                          : !llvm.vec<2 x half>, % arg10
                          : !llvm.vec<2 x half>, % arg11
                          : !llvm.vec<2 x half>, % arg12
                          : !llvm.vec<2 x half>, % arg13
                          : !llvm.vec<2 x half>, % arg14
                          : !llvm.vec<2 x half>, % arg15
                          : !llvm.vec<2 x half>, % arg16
                          : !llvm.vec<2 x half>, % arg17
                          : !llvm.vec<2 x half>, % arg18
                          : !llvm.vec<2 x half>, % arg19
                          : !llvm.vec<2 x half>) {
    %132 = nvvm.wmma.mma %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19{wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half> -> !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>

    llvm.return
  }
}
