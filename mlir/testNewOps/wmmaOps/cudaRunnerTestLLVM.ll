; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__wg_main_kernel_0 = internal addrspace(3) global [1024 x half] undef

declare i8* @malloc(i64)

declare void @free(i8*)

define void @main_kernel() !dbg !4 {
  %1 = alloca <2 x half>, i64 8, align 4, !dbg !8
  %2 = insertvalue { <2 x half>*, <2 x half>*, i64, [1 x i64], [1 x i64] } undef, <2 x half>* %1, 0, !dbg !10
  %3 = insertvalue { <2 x half>*, <2 x half>*, i64, [1 x i64], [1 x i64] } %2, <2 x half>* %1, 1, !dbg !11
  %4 = insertvalue { <2 x half>*, <2 x half>*, i64, [1 x i64], [1 x i64] } %3, i64 0, 2, !dbg !12
  %5 = insertvalue { <2 x half>*, <2 x half>*, i64, [1 x i64], [1 x i64] } %4, i64 8, 3, 0, !dbg !13
  %6 = insertvalue { <2 x half>*, <2 x half>*, i64, [1 x i64], [1 x i64] } %5, i64 1, 4, 0, !dbg !14
  br label %7, !dbg !15

7:                                                ; preds = %0
  %8 = extractvalue { <2 x half>*, <2 x half>*, i64, [1 x i64], [1 x i64] } %6, 0, !dbg !16
  %9 = extractvalue { <2 x half>*, <2 x half>*, i64, [1 x i64], [1 x i64] } %6, 1, !dbg !17
  %10 = extractvalue { <2 x half>*, <2 x half>*, i64, [1 x i64], [1 x i64] } %6, 2, !dbg !18
  %11 = extractvalue { <2 x half>*, <2 x half>*, i64, [1 x i64], [1 x i64] } %6, 3, 0, !dbg !19
  %12 = extractvalue { <2 x half>*, <2 x half>*, i64, [1 x i64], [1 x i64] } %6, 4, 0, !dbg !20
  %13 = call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p3i32(i32 addrspace(3)* bitcast (half addrspace(3)* getelementptr inbounds ([1024 x half], [1024 x half] addrspace(3)* @__wg_main_kernel_0, i32 0, i32 272) to i32 addrspace(3)*), i32 16), !dbg !21
  %14 = bitcast <2 x half>* %9 to i32*, !dbg !22
  %15 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %13, 0, !dbg !23
  ret void, !dbg !55
}

; Function Attrs: argmemonly nounwind readonly
declare { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p3i32(i32 addrspace(3)* nocapture readonly, i32) #0

attributes #0 = { argmemonly nounwind readonly }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!nvvm.annotations = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{void ()* @main_kernel, !"kernel", i32 1}
!4 = distinct !DISubprogram(name: "main_kernel", linkageName: "main_kernel", scope: null, file: !5, line: 2, type: !6, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !7)
!5 = !DIFile(filename: "cudaRunnerTestLLVM.ll", directory: "/home/navdeep/work/GPU_GEMM/llvm-project-mcl/mlir/testNewOps/wmmaOps")
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 20, column: 9, scope: !9)
!9 = !DILexicalBlockFile(scope: !4, file: !5, discriminator: 0)
!10 = !DILocation(line: 22, column: 9, scope: !9)
!11 = !DILocation(line: 23, column: 9, scope: !9)
!12 = !DILocation(line: 25, column: 9, scope: !9)
!13 = !DILocation(line: 27, column: 9, scope: !9)
!14 = !DILocation(line: 29, column: 9, scope: !9)
!15 = !DILocation(line: 30, column: 3, scope: !9)
!16 = !DILocation(line: 39, column: 9, scope: !9)
!17 = !DILocation(line: 40, column: 9, scope: !9)
!18 = !DILocation(line: 41, column: 9, scope: !9)
!19 = !DILocation(line: 42, column: 9, scope: !9)
!20 = !DILocation(line: 43, column: 9, scope: !9)
!21 = !DILocation(line: 51, column: 9, scope: !9)
!22 = !DILocation(line: 52, column: 9, scope: !9)
!23 = !DILocation(line: 53, column: 9, scope: !9)
!24 = !DILocation(line: 54, column: 9, scope: !9)
!25 = !DILocation(line: 56, column: 9, scope: !9)
!26 = !DILocation(line: 57, column: 3, scope: !9)
!27 = !DILocation(line: 58, column: 9, scope: !9)
!28 = !DILocation(line: 59, column: 9, scope: !9)
!29 = !DILocation(line: 61, column: 9, scope: !9)
!30 = !DILocation(line: 62, column: 3, scope: !9)
!31 = !DILocation(line: 63, column: 9, scope: !9)
!32 = !DILocation(line: 64, column: 9, scope: !9)
!33 = !DILocation(line: 66, column: 9, scope: !9)
!34 = !DILocation(line: 67, column: 3, scope: !9)
!35 = !DILocation(line: 68, column: 9, scope: !9)
!36 = !DILocation(line: 69, column: 9, scope: !9)
!37 = !DILocation(line: 71, column: 9, scope: !9)
!38 = !DILocation(line: 72, column: 3, scope: !9)
!39 = !DILocation(line: 73, column: 9, scope: !9)
!40 = !DILocation(line: 74, column: 9, scope: !9)
!41 = !DILocation(line: 76, column: 9, scope: !9)
!42 = !DILocation(line: 77, column: 3, scope: !9)
!43 = !DILocation(line: 78, column: 9, scope: !9)
!44 = !DILocation(line: 79, column: 9, scope: !9)
!45 = !DILocation(line: 81, column: 9, scope: !9)
!46 = !DILocation(line: 82, column: 3, scope: !9)
!47 = !DILocation(line: 83, column: 9, scope: !9)
!48 = !DILocation(line: 84, column: 9, scope: !9)
!49 = !DILocation(line: 86, column: 9, scope: !9)
!50 = !DILocation(line: 87, column: 3, scope: !9)
!51 = !DILocation(line: 88, column: 9, scope: !9)
!52 = !DILocation(line: 89, column: 9, scope: !9)
!53 = !DILocation(line: 91, column: 9, scope: !9)
!54 = !DILocation(line: 92, column: 3, scope: !9)
!55 = !DILocation(line: 93, column: 3, scope: !9)

