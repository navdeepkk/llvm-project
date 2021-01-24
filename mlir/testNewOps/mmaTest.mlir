llvm.func @nvvm_mma(%a0 : !llvm.vec<2 x half>, %a1 : !llvm.vec<2 x half>,
                    %b0 : !llvm.vec<2 x half>, %b1 : !llvm.vec<2 x half>,
                    %c0 : !llvm.float, %c1 : !llvm.float, %c2 : !llvm.float, %c3 : !llvm.float,
                    %c4 : !llvm.float, %c5 : !llvm.float, %c6 : !llvm.float, %c7 : !llvm.float) {
  // CHECK: call { float, float, float, float, float, float, float, float } @llvm.nvvm.mma.m8n8k4.row.col.f32.f32
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="row", blayout="col"} : (!llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float) -> !llvm.struct<(float, float, float, float, float, float, float, float)>
  llvm.return %0 : !llvm.struct<(float, float, float, float, float, float, float, float)>
}
