module attributes {llvm.data_layout = ""}  {
  llvm.func @simple_add1_add2_test(%arg0: !llvm.ptr<float>, %arg1: !llvm.ptr<float>) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.mlir.constant(0 : index) : !llvm.i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.mlir.constant(2 : index) : !llvm.i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.mlir.constant(1 : index) : !llvm.i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg1, %9[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg1, %10[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.mlir.constant(0 : index) : !llvm.i64
    %13 = llvm.insertvalue %12, %11[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.mlir.constant(2 : index) : !llvm.i64
    %15 = llvm.insertvalue %14, %13[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.mlir.constant(1 : index) : !llvm.i64
    %17 = llvm.insertvalue %16, %15[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.mlir.constant(2 : index) : !llvm.i64
    %19 = llvm.mlir.constant(0 : index) : !llvm.i64
    %20 = llvm.mlir.constant(1 : index) : !llvm.i64
    %21 = llvm.mlir.constant(1.000000e+00 : f32) : !llvm.float
    %22 = llvm.mlir.constant(2.000000e+00 : f32) : !llvm.float
    llvm.br ^bb1(%19 : !llvm.i64)
  ^bb1(%23: !llvm.i64):  // 2 preds: ^bb0, ^bb2
    %24 = llvm.icmp "slt" %23, %18 : !llvm.i64
    llvm.cond_br %24, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %25 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = llvm.getelementptr %25[%23] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %27 = llvm.load %26 : !llvm.ptr<float>
    %28 = llvm.fadd %27, %21 : !llvm.float
    %29 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %30 = llvm.getelementptr %29[%23] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    llvm.store %28, %30 : !llvm.ptr<float>
    %31 = llvm.extractvalue %17[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %32 = llvm.getelementptr %31[%23] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %33 = llvm.load %32 : !llvm.ptr<float>
    %34 = llvm.fadd %28, %22 : !llvm.float
    %35 = llvm.extractvalue %17[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %36 = llvm.getelementptr %35[%23] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    llvm.store %34, %36 : !llvm.ptr<float>
    %37 = llvm.add %23, %20 : !llvm.i64
    llvm.br ^bb1(%37 : !llvm.i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @malloc(!llvm.i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
  llvm.func @printF32(!llvm.float) attributes {sym_visibility = "private"}
  llvm.func @printComma() attributes {sym_visibility = "private"}
  llvm.func @printNewline() attributes {sym_visibility = "private"}
  llvm.func @main() {
    %0 = llvm.mlir.constant(2 : index) : !llvm.i64
    %1 = llvm.mlir.constant(0 : index) : !llvm.i64
    %2 = llvm.mlir.constant(1 : index) : !llvm.i64
    %3 = llvm.mlir.constant(1.000000e+00 : f32) : !llvm.float
    %4 = llvm.mlir.constant(2.000000e+00 : f32) : !llvm.float
    %5 = llvm.mlir.constant(2 : index) : !llvm.i64
    %6 = llvm.mlir.constant(1 : index) : !llvm.i64
    %7 = llvm.mlir.null : !llvm.ptr<float>
    %8 = llvm.getelementptr %7[%5] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %9 = llvm.ptrtoint %8 : !llvm.ptr<float> to !llvm.i64
    %10 = llvm.call @malloc(%9) : (!llvm.i64) -> !llvm.ptr<i8>
    %11 = llvm.bitcast %10 : !llvm.ptr<i8> to !llvm.ptr<float>
    %12 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.insertvalue %11, %12[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.insertvalue %11, %13[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.mlir.constant(0 : index) : !llvm.i64
    %16 = llvm.insertvalue %15, %14[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.insertvalue %5, %16[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.insertvalue %6, %17[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.mlir.constant(2 : index) : !llvm.i64
    %20 = llvm.mlir.constant(1 : index) : !llvm.i64
    %21 = llvm.mlir.null : !llvm.ptr<float>
    %22 = llvm.getelementptr %21[%19] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %23 = llvm.ptrtoint %22 : !llvm.ptr<float> to !llvm.i64
    %24 = llvm.call @malloc(%23) : (!llvm.i64) -> !llvm.ptr<i8>
    %25 = llvm.bitcast %24 : !llvm.ptr<i8> to !llvm.ptr<float>
    %26 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %27 = llvm.insertvalue %25, %26[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.insertvalue %25, %27[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %29 = llvm.mlir.constant(0 : index) : !llvm.i64
    %30 = llvm.insertvalue %29, %28[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.insertvalue %19, %30[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %32 = llvm.insertvalue %20, %31[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.br ^bb1(%1 : !llvm.i64)
  ^bb1(%33: !llvm.i64):  // 2 preds: ^bb0, ^bb2
    %34 = llvm.icmp "slt" %33, %0 : !llvm.i64
    llvm.cond_br %34, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %35 = llvm.extractvalue %18[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %36 = llvm.getelementptr %35[%33] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    llvm.store %3, %36 : !llvm.ptr<float>
    %37 = llvm.extractvalue %32[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %38 = llvm.getelementptr %37[%33] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    llvm.store %3, %38 : !llvm.ptr<float>
    %39 = llvm.add %33, %2 : !llvm.i64
    llvm.br ^bb1(%39 : !llvm.i64)
  ^bb3:  // pred: ^bb1
    %40 = llvm.extractvalue %18[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.extractvalue %32[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @simple_add1_add2_test(%40, %41) : (!llvm.ptr<float>, !llvm.ptr<float>) -> ()
    %42 = llvm.extractvalue %18[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %43 = llvm.getelementptr %42[%1] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %44 = llvm.load %43 : !llvm.ptr<float>
    llvm.call @printF32(%44) : (!llvm.float) -> ()
    llvm.call @printComma() : () -> ()
    %45 = llvm.extractvalue %18[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %46 = llvm.getelementptr %45[%2] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %47 = llvm.load %46 : !llvm.ptr<float>
    llvm.call @printF32(%47) : (!llvm.float) -> ()
    llvm.call @printNewline() : () -> ()
    %48 = llvm.extractvalue %32[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %49 = llvm.getelementptr %48[%1] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %50 = llvm.load %49 : !llvm.ptr<float>
    llvm.call @printF32(%50) : (!llvm.float) -> ()
    llvm.call @printComma() : () -> ()
    %51 = llvm.extractvalue %32[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %52 = llvm.getelementptr %51[%2] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %53 = llvm.load %52 : !llvm.ptr<float>
    llvm.call @printF32(%53) : (!llvm.float) -> ()
    llvm.call @printNewline() : () -> ()
    %54 = llvm.extractvalue %18[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %55 = llvm.bitcast %54 : !llvm.ptr<float> to !llvm.ptr<i8>
    llvm.call @free(%55) : (!llvm.ptr<i8>) -> ()
    %56 = llvm.extractvalue %32[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %57 = llvm.bitcast %56 : !llvm.ptr<float> to !llvm.ptr<i8>
    llvm.call @free(%57) : (!llvm.ptr<i8>) -> ()
    llvm.return
  }
}
