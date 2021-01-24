; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"

@__wg_main_kernel_0 = internal addrspace(3) global [32 x i32] undef

declare i8* @malloc(i64 %0)

declare void @free(i8* %0)

; External declaration of the puts function
declare i32 @puts(i8* nocapture) nounwind

define void @main_kernel(i32* %0, i32* %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i32* %7, i32* %8, i64 %9, i64 %10, i64 %11) {
  %13 = insertvalue { i32*, i32*, i64, [2 x i64], [2 x i64] } undef, i32* %0, 0
  %14 = insertvalue { i32*, i32*, i64, [2 x i64], [2 x i64] } %13, i32* %1, 1
  %15 = insertvalue { i32*, i32*, i64, [2 x i64], [2 x i64] } %14, i64 %2, 2
  %16 = insertvalue { i32*, i32*, i64, [2 x i64], [2 x i64] } %15, i64 %3, 3, 0
  %17 = insertvalue { i32*, i32*, i64, [2 x i64], [2 x i64] } %16, i64 %5, 4, 0
  %18 = insertvalue { i32*, i32*, i64, [2 x i64], [2 x i64] } %17, i64 %4, 3, 1
  %19 = insertvalue { i32*, i32*, i64, [2 x i64], [2 x i64] } %18, i64 %6, 4, 1
  %20 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } undef, i32* %7, 0
  %21 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %20, i32* %8, 1
  %22 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %21, i64 %9, 2
  %23 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %22, i64 %10, 3, 0
  %24 = insertvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %23, i64 %11, 4, 0
  %25 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %26 = sext i32 %25 to i64
  %27 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %28 = sext i32 %27 to i64
  br label %29

29:                                               ; preds = %12
  %30 = extractvalue { i32*, i32*, i64, [2 x i64], [2 x i64] } %19, 1
  %31 = mul i64 %26, 6
  %32 = add i64 %31, %28
  %33 = getelementptr i32, i32* %30, i64 %32
  %34 = load i32, i32* %33, align 4
  %35 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %36 = sext i32 %35 to i64
  %37 = trunc i64 %36 to i32
  %38 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %39 = sext i32 %38 to i64
  %40 = trunc i64 %39 to i32
  %41 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  %42 = sext i32 %41 to i64
  %43 = trunc i64 %42 to i32
  %44 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %45 = sext i32 %44 to i64
  %46 = trunc i64 %45 to i32
  %47 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %48 = sext i32 %47 to i64
  %49 = trunc i64 %48 to i32
  %50 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %51 = sext i32 %50 to i64
  %52 = trunc i64 %51 to i32
  %53 = mul i32 %52, %40
  %54 = add i32 %53, %49
  %55 = mul i32 %54, %37
  %56 = mul i32 %37, %40
  %57 = add i32 %55, %46
  %58 = mul i32 %56, %43
  %59 = and i32 %57, 31
  %60 = icmp eq i32 %59, 0
  %61 = sub i32 %57, %59
  %62 = sub i32 %58, %61
  %63 = icmp slt i32 %62, 32
  br i1 %63, label %64, label %125

64:                                               ; preds = %29
  %65 = shl i32 1, %62
  %66 = sub i32 %65, 1
  %67 = sub i32 %62, 1
  %68 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 %66, i32 %34, i32 1, i32 %67)
  %69 = extractvalue { i32, i1 } %68, 0
  %70 = extractvalue { i32, i1 } %68, 1
  br i1 %70, label %71, label %74

71:                                               ; preds = %64
  %72 = icmp ugt i32 %34, %69
  %73 = select i1 %72, i32 %34, i32 %69
  br label %75

74:                                               ; preds = %64
  br label %75

75:                                               ; preds = %71, %74
  %76 = phi i32 [ %73, %71 ], [ %34, %74 ]
  %77 = shl i32 1, %62
  %78 = sub i32 %77, 1
  %79 = sub i32 %62, 1
  %80 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 %78, i32 %76, i32 2, i32 %79)
  %81 = extractvalue { i32, i1 } %80, 0
  %82 = extractvalue { i32, i1 } %80, 1
  br i1 %82, label %83, label %86

83:                                               ; preds = %75
  %84 = icmp ugt i32 %76, %81
  %85 = select i1 %84, i32 %76, i32 %81
  br label %87

86:                                               ; preds = %75
  br label %87

87:                                               ; preds = %83, %86
  %88 = phi i32 [ %85, %83 ], [ %76, %86 ]
  %89 = shl i32 1, %62
  %90 = sub i32 %89, 1
  %91 = sub i32 %62, 1
  %92 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 %90, i32 %88, i32 4, i32 %91)
  %93 = extractvalue { i32, i1 } %92, 0
  %94 = extractvalue { i32, i1 } %92, 1
  br i1 %94, label %95, label %98

95:                                               ; preds = %87
  %96 = icmp ugt i32 %88, %93
  %97 = select i1 %96, i32 %88, i32 %93
  br label %99

98:                                               ; preds = %87
  br label %99

99:                                               ; preds = %95, %98
  %100 = phi i32 [ %97, %95 ], [ %88, %98 ]
  %101 = shl i32 1, %62
  %102 = sub i32 %101, 1
  %103 = sub i32 %62, 1
  %104 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 %102, i32 %100, i32 8, i32 %103)
  %105 = extractvalue { i32, i1 } %104, 0
  %106 = extractvalue { i32, i1 } %104, 1
  br i1 %106, label %107, label %110

107:                                              ; preds = %99
  %108 = icmp ugt i32 %100, %105
  %109 = select i1 %108, i32 %100, i32 %105
  br label %111

110:                                              ; preds = %99
  br label %111

111:                                              ; preds = %107, %110
  %112 = phi i32 [ %109, %107 ], [ %100, %110 ]
  %113 = shl i32 1, %62
  %114 = sub i32 %113, 1
  %115 = sub i32 %62, 1
  %116 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 %114, i32 %112, i32 16, i32 %115)
  %117 = extractvalue { i32, i1 } %116, 0
  %118 = extractvalue { i32, i1 } %116, 1
  br i1 %118, label %119, label %122

119:                                              ; preds = %111
  %120 = icmp ugt i32 %112, %117
  %121 = select i1 %120, i32 %112, i32 %117
  br label %123

122:                                              ; preds = %111
  br label %123

123:                                              ; preds = %119, %122
  %124 = phi i32 [ %121, %119 ], [ %112, %122 ]
  br label %151

125:                                              ; preds = %29
  %126 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 poison, i32 %34, i32 1, i32 31)
  %127 = extractvalue { i32, i1 } %126, 0
  %128 = extractvalue { i32, i1 } %126, 1
  %129 = icmp ugt i32 %34, %127
  %130 = select i1 %129, i32 %34, i32 %127
  %131 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 poison, i32 %130, i32 2, i32 31)
  %132 = extractvalue { i32, i1 } %131, 0
  %133 = extractvalue { i32, i1 } %131, 1
  %134 = icmp ugt i32 %130, %132
  %135 = select i1 %134, i32 %130, i32 %132
  %136 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 poison, i32 %135, i32 4, i32 31)
  %137 = extractvalue { i32, i1 } %136, 0
  %138 = extractvalue { i32, i1 } %136, 1
  %139 = icmp ugt i32 %135, %137
  %140 = select i1 %139, i32 %135, i32 %137
  %141 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 poison, i32 %140, i32 8, i32 31)
  %142 = extractvalue { i32, i1 } %141, 0
  %143 = extractvalue { i32, i1 } %141, 1
  %144 = icmp ugt i32 %140, %142
  %145 = select i1 %144, i32 %140, i32 %142
  %146 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 poison, i32 %145, i32 16, i32 31)
  %147 = extractvalue { i32, i1 } %146, 0
  %148 = extractvalue { i32, i1 } %146, 1
  %149 = icmp ugt i32 %145, %147
  %150 = select i1 %149, i32 %145, i32 %147
  br label %151

151:                                              ; preds = %123, %125
  %152 = phi i32 [ %124, %123 ], [ %150, %125 ]
  br i1 %60, label %153, label %157

153:                                              ; preds = %151
  %154 = sdiv i32 %57, 32
  %155 = sext i32 %154 to i64
  %156 = getelementptr i32, i32 addrspace(3)* getelementptr inbounds ([32 x i32], [32 x i32] addrspace(3)* @__wg_main_kernel_0, i32 0, i32 0), i64 %155
  store i32 %152, i32 addrspace(3)* %156, align 4
  br label %158

157:                                              ; preds = %151
  br label %158

158:                                              ; preds = %153, %157
  call void @llvm.nvvm.barrier0()
  %159 = add i32 %58, 31
  %160 = sdiv i32 %159, 32
  %161 = icmp slt i32 %57, %160
  br i1 %161, label %162, label %256

162:                                              ; preds = %158
  %163 = sext i32 %57 to i64
  %164 = getelementptr i32, i32 addrspace(3)* getelementptr inbounds ([32 x i32], [32 x i32] addrspace(3)* @__wg_main_kernel_0, i32 0, i32 0), i64 %163
  %165 = load i32, i32 addrspace(3)* %164, align 4
  %166 = icmp slt i32 %160, 32
  br i1 %166, label %167, label %228

167:                                              ; preds = %162
  %168 = shl i32 1, %160
  %169 = sub i32 %168, 1
  %170 = sub i32 %160, 1
  %171 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 %169, i32 %165, i32 1, i32 %170)
  %172 = extractvalue { i32, i1 } %171, 0
  %173 = extractvalue { i32, i1 } %171, 1
  br i1 %173, label %174, label %177

174:                                              ; preds = %167
  %175 = icmp ugt i32 %165, %172
  %176 = select i1 %175, i32 %165, i32 %172
  br label %178

177:                                              ; preds = %167
  br label %178

178:                                              ; preds = %174, %177
  %179 = phi i32 [ %176, %174 ], [ %165, %177 ]
  %180 = shl i32 1, %160
  %181 = sub i32 %180, 1
  %182 = sub i32 %160, 1
  %183 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 %181, i32 %179, i32 2, i32 %182)
  %184 = extractvalue { i32, i1 } %183, 0
  %185 = extractvalue { i32, i1 } %183, 1
  br i1 %185, label %186, label %189

186:                                              ; preds = %178
  %187 = icmp ugt i32 %179, %184
  %188 = select i1 %187, i32 %179, i32 %184
  br label %190

189:                                              ; preds = %178
  br label %190

190:                                              ; preds = %186, %189
  %191 = phi i32 [ %188, %186 ], [ %179, %189 ]
  %192 = shl i32 1, %160
  %193 = sub i32 %192, 1
  %194 = sub i32 %160, 1
  %195 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 %193, i32 %191, i32 4, i32 %194)
  %196 = extractvalue { i32, i1 } %195, 0
  %197 = extractvalue { i32, i1 } %195, 1
  br i1 %197, label %198, label %201

198:                                              ; preds = %190
  %199 = icmp ugt i32 %191, %196
  %200 = select i1 %199, i32 %191, i32 %196
  br label %202

201:                                              ; preds = %190
  br label %202

202:                                              ; preds = %198, %201
  %203 = phi i32 [ %200, %198 ], [ %191, %201 ]
  %204 = shl i32 1, %160
  %205 = sub i32 %204, 1
  %206 = sub i32 %160, 1
  %207 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 %205, i32 %203, i32 8, i32 %206)
  %208 = extractvalue { i32, i1 } %207, 0
  %209 = extractvalue { i32, i1 } %207, 1
  br i1 %209, label %210, label %213

210:                                              ; preds = %202
  %211 = icmp ugt i32 %203, %208
  %212 = select i1 %211, i32 %203, i32 %208
  br label %214

213:                                              ; preds = %202
  br label %214

214:                                              ; preds = %210, %213
  %215 = phi i32 [ %212, %210 ], [ %203, %213 ]
  %216 = shl i32 1, %160
  %217 = sub i32 %216, 1
  %218 = sub i32 %160, 1
  %219 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 %217, i32 %215, i32 16, i32 %218)
  %220 = extractvalue { i32, i1 } %219, 0
  %221 = extractvalue { i32, i1 } %219, 1
  br i1 %221, label %222, label %225

222:                                              ; preds = %214
  %223 = icmp ugt i32 %215, %220
  %224 = select i1 %223, i32 %215, i32 %220
  br label %226

225:                                              ; preds = %214
  br label %226

226:                                              ; preds = %222, %225
  %227 = phi i32 [ %224, %222 ], [ %215, %225 ]
  br label %254

228:                                              ; preds = %162
  %229 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 poison, i32 %165, i32 1, i32 31)
  %230 = extractvalue { i32, i1 } %229, 0
  %231 = extractvalue { i32, i1 } %229, 1
  %232 = icmp ugt i32 %165, %230
  %233 = select i1 %232, i32 %165, i32 %230
  %234 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 poison, i32 %233, i32 2, i32 31)
  %235 = extractvalue { i32, i1 } %234, 0
  %236 = extractvalue { i32, i1 } %234, 1
  %237 = icmp ugt i32 %233, %235
  %238 = select i1 %237, i32 %233, i32 %235
  %239 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 poison, i32 %238, i32 4, i32 31)
  %240 = extractvalue { i32, i1 } %239, 0
  %241 = extractvalue { i32, i1 } %239, 1
  %242 = icmp ugt i32 %238, %240
  %243 = select i1 %242, i32 %238, i32 %240
  %244 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 poison, i32 %243, i32 8, i32 31)
  %245 = extractvalue { i32, i1 } %244, 0
  %246 = extractvalue { i32, i1 } %244, 1
  %247 = icmp ugt i32 %243, %245
  %248 = select i1 %247, i32 %243, i32 %245
  %249 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 poison, i32 %248, i32 16, i32 31)
  %250 = extractvalue { i32, i1 } %249, 0
  %251 = extractvalue { i32, i1 } %249, 1
  %252 = icmp ugt i32 %248, %250
  %253 = select i1 %252, i32 %248, i32 %250
  br label %254

254:                                              ; preds = %226, %228
  %255 = phi i32 [ %227, %226 ], [ %253, %228 ]
  store i32 %255, i32 addrspace(3)* getelementptr inbounds ([32 x i32], [32 x i32] addrspace(3)* @__wg_main_kernel_0, i32 0, i32 0), align 4
  br label %257

256:                                              ; preds = %158
  br label %257

257:                                              ; preds = %254, %256
  call void @llvm.nvvm.barrier0()
  %258 = load i32, i32 addrspace(3)* getelementptr inbounds ([32 x i32], [32 x i32] addrspace(3)* @__wg_main_kernel_0, i32 0, i32 0), align 4
  %259 = extractvalue { i32*, i32*, i64, [1 x i64], [1 x i64] } %24, 1
  %260 = getelementptr i32, i32* %259, i64 %26
  store i32 %258, i32* %260, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0

; Function Attrs: convergent inaccessiblememonly nounwind
declare { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 %0, i32 %1, i32 %2, i32 %3) #1

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #2

attributes #0 = { nounwind readnone }
attributes #1 = { convergent inaccessiblememonly nounwind }
attributes #2 = { convergent nounwind }

!nvvm.annotations = !{!0}

!0 = !{void (i32*, i32*, i64, i64, i64, i64, i64, i32*, i32*, i64, i64, i64)* @main_kernel, !"kernel", i32 1}
