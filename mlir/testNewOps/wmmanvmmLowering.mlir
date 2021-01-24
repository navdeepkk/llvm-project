llvm.func @nvvm_wmma_load(%p : !llvm.ptr<ptr<half>>) {
  %0 = nvvm.wmma.load %p {layout = "RowMajor", operand = "AOp", srcMemSpace = 3 : ui8, stride = 16 : ui16, wk = 16 : ui16, wm = 16 : ui16, wn = 16 : ui16} : (!llvm.ptr<ptr<half>>) -> !llvm.vec<2 x half>
  llvm.return %0 : !llvm.vec<2xhalf>
}
