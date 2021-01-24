module attributes {llvm.data_layout = ""}  {
  llvm.func @simple_add1_add2_test(%arg0: !llvm.ptr<float>, %arg1: !llvm.ptr<float>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.ptr<float>, %arg6: !llvm.ptr<float>, %arg7: !llvm.i64, %arg8: !llvm.i64, %arg9: !llvm.i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg9, %10[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.mlir.constant(2 : index) : !llvm.i64
    %13 = llvm.mlir.constant(0 : index) : !llvm.i64
    %14 = llvm.mlir.constant(1 : index) : !llvm.i64
    %15 = llvm.mlir.constant(1.000000e+00 : f32) : !llvm.float
    %16 = llvm.mlir.constant(2.000000e+00 : f32) : !llvm.float
    llvm.br ^bb1(%13 : !llvm.i64)
  ^bb1(%17: !llvm.i64):  // 2 preds: ^bb0, ^bb2
    %18 = llvm.icmp "slt" %17, %12 : !llvm.i64
    llvm.cond_br %18, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %19 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.getelementptr %19[%17] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %21 = llvm.load %20 : !llvm.ptr<float>
    %22 = llvm.fadd %21, %15 : !llvm.float
    %23 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.getelementptr %23[%17] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    llvm.store %22, %24 : !llvm.ptr<float>
    %25 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = llvm.getelementptr %25[%17] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %27 = llvm.load %26 : !llvm.ptr<float>
    %28 = llvm.fadd %22, %16 : !llvm.float
    %29 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %30 = llvm.getelementptr %29[%17] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    llvm.store %28, %30 : !llvm.ptr<float>
    %31 = llvm.add %17, %14 : !llvm.i64
    llvm.br ^bb1(%31 : !llvm.i64)
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
    %40 = llvm.extractvalue %18[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.extractvalue %18[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %42 = llvm.extractvalue %18[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %43 = llvm.extractvalue %18[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %44 = llvm.extractvalue %18[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %45 = llvm.extractvalue %32[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %46 = llvm.extractvalue %32[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %47 = llvm.extractvalue %32[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %48 = llvm.extractvalue %32[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %49 = llvm.extractvalue %32[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @simple_add1_add2_test(%40, %41, %42, %43, %44, %45, %46, %47, %48, %49) : (!llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    %50 = llvm.extractvalue %18[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %51 = llvm.getelementptr %50[%1] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %52 = llvm.load %51 : !llvm.ptr<float>
    llvm.call @printF32(%52) : (!llvm.float) -> ()
    llvm.call @printComma() : () -> ()
    %53 = llvm.extractvalue %18[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %54 = llvm.getelementptr %53[%2] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %55 = llvm.load %54 : !llvm.ptr<float>
    llvm.call @printF32(%55) : (!llvm.float) -> ()
    llvm.call @printNewline() : () -> ()
    %56 = llvm.extractvalue %32[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %57 = llvm.getelementptr %56[%1] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %58 = llvm.load %57 : !llvm.ptr<float>
    llvm.call @printF32(%58) : (!llvm.float) -> ()
    llvm.call @printComma() : () -> ()
    %59 = llvm.extractvalue %32[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %60 = llvm.getelementptr %59[%2] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %61 = llvm.load %60 : !llvm.ptr<float>
    llvm.call @printF32(%61) : (!llvm.float) -> ()
    llvm.call @printNewline() : () -> ()
    %62 = llvm.extractvalue %18[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %63 = llvm.bitcast %62 : !llvm.ptr<float> to !llvm.ptr<i8>
    llvm.call @free(%63) : (!llvm.ptr<i8>) -> ()
    %64 = llvm.extractvalue %32[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
    %65 = llvm.bitcast %64 : !llvm.ptr<float> to !llvm.ptr<i8>
    llvm.call @free(%65) : (!llvm.ptr<i8>) -> ()
    llvm.return
  }
}

