module attributes {llvm.data_layout = ""}  {
  llvm.func @malloc(!llvm.i64) -> !llvm.ptr<i8>
  llvm.func @matmul() {
    %0 = llvm.mlir.constant(8 : index) : !llvm.i64
    %1 = llvm.mlir.constant(8 : index) : !llvm.i64
    %2 = llvm.mlir.constant(1 : index) : !llvm.i64
    %3 = llvm.mlir.constant(64 : index) : !llvm.i64
    %4 = llvm.mlir.null : !llvm.ptr<vec<16 x half>>
    %5 = llvm.getelementptr %4[%3] : (!llvm.ptr<vec<16 x half>>, !llvm.i64) -> !llvm.ptr<vec<16 x half>>
    %6 = llvm.ptrtoint %5 : !llvm.ptr<vec<16 x half>> to !llvm.i64
    %7 = llvm.mlir.null : !llvm.ptr<vec<16 x half>>
    %8 = llvm.mlir.constant(1 : index) : !llvm.i64
    %9 = llvm.getelementptr %7[%8] : (!llvm.ptr<vec<16 x half>>, !llvm.i64) -> !llvm.ptr<vec<16 x half>>
    %10 = llvm.ptrtoint %9 : !llvm.ptr<vec<16 x half>> to !llvm.i64
    %11 = llvm.add %6, %10 : !llvm.i64
    %12 = llvm.call @malloc(%11) : (!llvm.i64) -> !llvm.ptr<i8>
    %13 = llvm.bitcast %12 : !llvm.ptr<i8> to !llvm.ptr<vec<16 x half>>
    %14 = llvm.ptrtoint %13 : !llvm.ptr<vec<16 x half>> to !llvm.i64
    %15 = llvm.mlir.constant(1 : index) : !llvm.i64
    %16 = llvm.sub %10, %15 : !llvm.i64
    %17 = llvm.add %14, %16 : !llvm.i64
    %18 = llvm.urem %17, %10 : !llvm.i64
    %19 = llvm.sub %17, %18 : !llvm.i64
    %20 = llvm.inttoptr %19 : !llvm.i64 to !llvm.ptr<vec<16 x half>>
    %21 = llvm.mlir.undef : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.insertvalue %13, %21[0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<2 x i64>, array<2 x i64>)>
    %23 = llvm.insertvalue %20, %22[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.mlir.constant(0 : index) : !llvm.i64
    %25 = llvm.insertvalue %24, %23[2] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<2 x i64>, array<2 x i64>)>
    %26 = llvm.insertvalue %0, %25[3, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<2 x i64>, array<2 x i64>)>
    %27 = llvm.insertvalue %1, %26[3, 1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<2 x i64>, array<2 x i64>)>
    %28 = llvm.insertvalue %1, %27[4, 0] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<2 x i64>, array<2 x i64>)>
    %29 = llvm.insertvalue %2, %28[4, 1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<2 x i64>, array<2 x i64>)>
    %30 = llvm.mlir.constant(0 : index) : !llvm.i64
    %31 = llvm.extractvalue %29[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.mlir.constant(8 : index) : !llvm.i64
    %33 = llvm.mul %30, %32 : !llvm.i64
    %34 = llvm.add %33, %30 : !llvm.i64
    %35 = llvm.getelementptr %31[%34] : (!llvm.ptr<vec<16 x half>>, !llvm.i64) -> !llvm.ptr<vec<16 x half>>
    %36 = llvm.load %35 : !llvm.ptr<vec<16 x half>>
    %37 = llvm.extractvalue %29[1] : !llvm.struct<(ptr<vec<16 x half>>, ptr<vec<16 x half>>, i64, array<2 x i64>, array<2 x i64>)>
    %38 = llvm.mlir.constant(8 : index) : !llvm.i64
    %39 = llvm.mul %30, %38 : !llvm.i64
    %40 = llvm.add %39, %30 : !llvm.i64
    %41 = llvm.getelementptr %37[%40] : (!llvm.ptr<vec<16 x half>>, !llvm.i64) -> !llvm.ptr<vec<16 x half>>
    llvm.store %36, %41 : !llvm.ptr<vec<16 x half>>
    llvm.return
  }
}

