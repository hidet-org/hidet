; ModuleID = 'TVMMod'
source_filename = "TVMMod"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%0 = type { i32*, i32 }
%1 = type { i8*, %2, i32, %3, i64*, i64*, i64 }
%2 = type { i32, i32 }
%3 = type { i8, i8, i16 }
%closure_loop_parallel_ax0.ax1.fused = type { i8*, i8* }
%closure_loop_parallel_x.outer.y.outer.fused = type { i8*, i8*, i8* }

@__TVMAPISetLastError = linkonce dllexport local_unnamed_addr global void (i8*)* null, align 8
@__TVMBackendParallelLaunch = linkonce dllexport local_unnamed_addr global i32 (i32 (i32, %0*, i8*)*, i8*, i32)* null, align 8
@.str = private constant [88 x i8] c"Assert fail: num_args == 2, tvmgen_default_fused_layout_transform: num_args should be 2\00", align 1
@.str.1 = private constant [159 x i8] c"Assert fail: arg_p0_code == 3 or arg_p0_code == 13 or arg_p0_code == 7 or arg_p0_code == 4, tvmgen_default_fused_layout_transform: Expect arg[0] to be pointer\00", align 1
@.str.2 = private constant [207 x i8] c"Assert fail: arg_T_layout_trans_code == 3 or arg_T_layout_trans_code == 13 or arg_T_layout_trans_code == 7 or arg_T_layout_trans_code == 4, tvmgen_default_fused_layout_transform: Expect arg[1] to be pointer\00", align 1
@.str.3 = private constant [94 x i8] c"Assert fail: 2 == T.tvm_struct_get(arg_p0, 0, 4, \22int32\22), arg.p0.ndim is expected to equal 2\00", align 1
@.str.4 = private constant [226 x i8] c"Assert fail: T.tvm_struct_get(arg_p0, 0, 5, \22uint8\22) == T.uint8(2) and T.tvm_struct_get(arg_p0, 0, 6, \22uint8\22) == T.uint8(32) and T.tvm_struct_get(arg_p0, 0, 7, \22uint16\22) == T.uint16(1), arg.p0.dtype is expected to be float32\00", align 1
@.str.5 = private constant [150 x i8] c"Assert fail: T.Cast(\22int32\22, arg_p0_shape[0]) == 512, Argument arg.p0.shape[0] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, arg_p0_shape[0])\00", align 1
@.str.6 = private constant [150 x i8] c"Assert fail: T.Cast(\22int32\22, arg_p0_shape[1]) == 512, Argument arg.p0.shape[1] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, arg_p0_shape[1])\00", align 1
@.str.7 = private constant [145 x i8] c"Assert fail: 1 == T.Cast(\22int32\22, arg_p0_strides[1]) and 512 == T.Cast(\22int32\22, arg_p0_strides[0]), arg.p0.strides: expected to be compact array\00", align 1
@.str.8 = private constant [185 x i8] c"Assert fail: T.uint64(0) == T.tvm_struct_get(arg_p0, 0, 8, \22uint64\22), Argument arg.p0.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(arg_p0, 0, 8, \22uint64\22)\00", align 1
@.str.9 = private constant [165 x i8] c"Assert fail: T.tvm_struct_get(arg_p0, 0, 10, \22int32\22) == 1, Argument arg.p0.device_type has an unsatisfied constraint: 1 == T.tvm_struct_get(arg_p0, 0, 10, \22int32\22)\00", align 1
@.str.10 = private constant [118 x i8] c"Assert fail: 3 == T.tvm_struct_get(arg_T_layout_trans, 0, 4, \22int32\22), arg.T_layout_trans.ndim is expected to equal 3\00", align 1
@.str.11 = private constant [274 x i8] c"Assert fail: T.tvm_struct_get(arg_T_layout_trans, 0, 5, \22uint8\22) == T.uint8(2) and T.tvm_struct_get(arg_T_layout_trans, 0, 6, \22uint8\22) == T.uint8(32) and T.tvm_struct_get(arg_T_layout_trans, 0, 7, \22uint16\22) == T.uint16(1), arg.T_layout_trans.dtype is expected to be float32\00", align 1
@.str.12 = private constant [184 x i8] c"Assert fail: T.Cast(\22int32\22, arg_T_layout_trans_shape[0]) == 32, Argument arg.T_layout_trans.shape[0] has an unsatisfied constraint: 32 == T.Cast(\22int32\22, arg_T_layout_trans_shape[0])\00", align 1
@.str.13 = private constant [186 x i8] c"Assert fail: T.Cast(\22int32\22, arg_T_layout_trans_shape[1]) == 512, Argument arg.T_layout_trans.shape[1] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, arg_T_layout_trans_shape[1])\00", align 1
@.str.14 = private constant [184 x i8] c"Assert fail: T.Cast(\22int32\22, arg_T_layout_trans_shape[2]) == 16, Argument arg.T_layout_trans.shape[2] has an unsatisfied constraint: 16 == T.Cast(\22int32\22, arg_T_layout_trans_shape[2])\00", align 1
@.str.15 = private constant [239 x i8] c"Assert fail: 1 == T.Cast(\22int32\22, arg_T_layout_trans_strides[2]) and 16 == T.Cast(\22int32\22, arg_T_layout_trans_strides[1]) and 8192 == T.Cast(\22int32\22, arg_T_layout_trans_strides[0]), arg.T_layout_trans.strides: expected to be compact array\00", align 1
@.str.16 = private constant [221 x i8] c"Assert fail: T.uint64(0) == T.tvm_struct_get(arg_T_layout_trans, 0, 8, \22uint64\22), Argument arg.T_layout_trans.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(arg_T_layout_trans, 0, 8, \22uint64\22)\00", align 1
@.str.17 = private constant [201 x i8] c"Assert fail: T.tvm_struct_get(arg_T_layout_trans, 0, 10, \22int32\22) == 1, Argument arg.T_layout_trans.device_type has an unsatisfied constraint: 1 == T.tvm_struct_get(arg_T_layout_trans, 0, 10, \22int32\22)\00", align 1
@.str.18 = private constant [207 x i8] c"Assert fail: dev_id == T.tvm_struct_get(arg_T_layout_trans, 0, 9, \22int32\22), Argument arg.T_layout_trans.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(arg_T_layout_trans, 0, 9, \22int32\22)\00", align 1
@.str.19 = private constant [93 x i8] c"Assert fail: num_args == 3, tvmgen_default_fused_nn_contrib_dense_pack: num_args should be 3\00", align 1
@.str.20 = private constant [164 x i8] c"Assert fail: arg_p0_code == 3 or arg_p0_code == 13 or arg_p0_code == 7 or arg_p0_code == 4, tvmgen_default_fused_nn_contrib_dense_pack: Expect arg[0] to be pointer\00", align 1
@.str.21 = private constant [164 x i8] c"Assert fail: arg_p1_code == 3 or arg_p1_code == 13 or arg_p1_code == 7 or arg_p1_code == 4, tvmgen_default_fused_nn_contrib_dense_pack: Expect arg[1] to be pointer\00", align 1
@.str.22 = private constant [184 x i8] c"Assert fail: arg_compute_code == 3 or arg_compute_code == 13 or arg_compute_code == 7 or arg_compute_code == 4, tvmgen_default_fused_nn_contrib_dense_pack: Expect arg[2] to be pointer\00", align 1
@.str.23 = private constant [94 x i8] c"Assert fail: 3 == T.tvm_struct_get(arg_p1, 0, 4, \22int32\22), arg.p1.ndim is expected to equal 3\00", align 1
@.str.24 = private constant [226 x i8] c"Assert fail: T.tvm_struct_get(arg_p1, 0, 5, \22uint8\22) == T.uint8(2) and T.tvm_struct_get(arg_p1, 0, 6, \22uint8\22) == T.uint8(32) and T.tvm_struct_get(arg_p1, 0, 7, \22uint16\22) == T.uint16(1), arg.p1.dtype is expected to be float32\00", align 1
@.str.25 = private constant [148 x i8] c"Assert fail: T.Cast(\22int32\22, arg_p1_shape[0]) == 32, Argument arg.p1.shape[0] has an unsatisfied constraint: 32 == T.Cast(\22int32\22, arg_p1_shape[0])\00", align 1
@.str.26 = private constant [150 x i8] c"Assert fail: T.Cast(\22int32\22, arg_p1_shape[1]) == 512, Argument arg.p1.shape[1] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, arg_p1_shape[1])\00", align 1
@.str.27 = private constant [148 x i8] c"Assert fail: T.Cast(\22int32\22, arg_p1_shape[2]) == 16, Argument arg.p1.shape[2] has an unsatisfied constraint: 16 == T.Cast(\22int32\22, arg_p1_shape[2])\00", align 1
@.str.28 = private constant [191 x i8] c"Assert fail: 1 == T.Cast(\22int32\22, arg_p1_strides[2]) and 16 == T.Cast(\22int32\22, arg_p1_strides[1]) and 8192 == T.Cast(\22int32\22, arg_p1_strides[0]), arg.p1.strides: expected to be compact array\00", align 1
@.str.29 = private constant [185 x i8] c"Assert fail: T.uint64(0) == T.tvm_struct_get(arg_p1, 0, 8, \22uint64\22), Argument arg.p1.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(arg_p1, 0, 8, \22uint64\22)\00", align 1
@.str.30 = private constant [165 x i8] c"Assert fail: T.tvm_struct_get(arg_p1, 0, 10, \22int32\22) == 1, Argument arg.p1.device_type has an unsatisfied constraint: 1 == T.tvm_struct_get(arg_p1, 0, 10, \22int32\22)\00", align 1
@.str.31 = private constant [171 x i8] c"Assert fail: dev_id == T.tvm_struct_get(arg_p1, 0, 9, \22int32\22), Argument arg.p1.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(arg_p1, 0, 9, \22int32\22)\00", align 1
@.str.32 = private constant [104 x i8] c"Assert fail: 2 == T.tvm_struct_get(arg_compute, 0, 4, \22int32\22), arg.compute.ndim is expected to equal 2\00", align 1
@.str.33 = private constant [246 x i8] c"Assert fail: T.tvm_struct_get(arg_compute, 0, 5, \22uint8\22) == T.uint8(2) and T.tvm_struct_get(arg_compute, 0, 6, \22uint8\22) == T.uint8(32) and T.tvm_struct_get(arg_compute, 0, 7, \22uint16\22) == T.uint16(1), arg.compute.dtype is expected to be float32\00", align 1
@.str.34 = private constant [165 x i8] c"Assert fail: T.Cast(\22int32\22, arg_compute_shape[0]) == 512, Argument arg.compute.shape[0] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, arg_compute_shape[0])\00", align 1
@.str.35 = private constant [165 x i8] c"Assert fail: T.Cast(\22int32\22, arg_compute_shape[1]) == 512, Argument arg.compute.shape[1] has an unsatisfied constraint: 512 == T.Cast(\22int32\22, arg_compute_shape[1])\00", align 1
@.str.36 = private constant [160 x i8] c"Assert fail: 1 == T.Cast(\22int32\22, arg_compute_strides[1]) and 512 == T.Cast(\22int32\22, arg_compute_strides[0]), arg.compute.strides: expected to be compact array\00", align 1
@.str.37 = private constant [200 x i8] c"Assert fail: T.uint64(0) == T.tvm_struct_get(arg_compute, 0, 8, \22uint64\22), Argument arg.compute.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(arg_compute, 0, 8, \22uint64\22)\00", align 1
@.str.38 = private constant [180 x i8] c"Assert fail: T.tvm_struct_get(arg_compute, 0, 10, \22int32\22) == 1, Argument arg.compute.device_type has an unsatisfied constraint: 1 == T.tvm_struct_get(arg_compute, 0, 10, \22int32\22)\00", align 1
@.str.39 = private constant [186 x i8] c"Assert fail: dev_id == T.tvm_struct_get(arg_compute, 0, 9, \22int32\22), Argument arg.compute.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(arg_compute, 0, 9, \22int32\22)\00", align 1
@llvm.global_ctors = appending global [0 x { i32, void ()*, i8* }] zeroinitializer

define dllexport i32 @tvmgen_default_fused_layout_transform(i8* noalias nocapture readonly %args, i32* noalias nocapture readonly %arg_type_ids, i32 %num_args, i8* noalias nocapture readnone %out_ret_value, i32* noalias nocapture readnone %out_ret_tcode, i8* noalias nocapture readnone %resource_handle) local_unnamed_addr #0 !dbg !5 {
entry:
  call void @llvm.dbg.value(metadata i8* %args, metadata !12, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32* %arg_type_ids, metadata !13, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 %num_args, metadata !14, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i8* %out_ret_value, metadata !15, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32* %out_ret_tcode, metadata !16, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i8* %resource_handle, metadata !17, metadata !DIExpression()), !dbg !18
  %0 = icmp eq i32 %num_args, 2, !dbg !18
  br i1 %0, label %assert_end, label %assert_fail, !dbg !18, !prof !19

assert_fail:                                      ; preds = %entry
  %1 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %1(i8* getelementptr inbounds ([88 x i8], [88 x i8]* @.str, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_end:                                       ; preds = %entry
  %2 = bitcast i8* %args to %1**, !dbg !18
  %arg.p046 = load %1*, %1** %2, align 8, !dbg !18
  %arg.p0.code = load i32, i32* %arg_type_ids, align 4, !dbg !18, !tbaa !23
  %3 = getelementptr inbounds i8, i8* %args, i64 8, !dbg !18
  %4 = bitcast i8* %3 to %1**, !dbg !18
  %arg.T_layout_trans47 = load %1*, %1** %4, align 8, !dbg !18
  %5 = getelementptr inbounds i32, i32* %arg_type_ids, i64 1, !dbg !18
  %arg.T_layout_trans.code = load i32, i32* %5, align 4, !dbg !18, !tbaa !34
  %6 = getelementptr inbounds %1, %1* %arg.p046, i64 0, i32 0, !dbg !18
  %p0 = load i8*, i8** %6, align 8, !dbg !18
  %ptrint = ptrtoint i8* %p0 to i64, !dbg !18
  %maskedptr = and i64 %ptrint, 63, !dbg !18
  %maskcond = icmp eq i64 %maskedptr, 0, !dbg !18
  tail call void @llvm.assume(i1 %maskcond), !dbg !18
  %7 = getelementptr inbounds %1, %1* %arg.p046, i64 0, i32 4, !dbg !18
  %arg.p0.shape = load i64*, i64** %7, align 8, !dbg !18
  %8 = getelementptr inbounds %1, %1* %arg.p046, i64 0, i32 5, !dbg !18
  %arg.p0.strides = load i64*, i64** %8, align 8, !dbg !18
  %9 = getelementptr inbounds %1, %1* %arg.p046, i64 0, i32 1, i32 1, !dbg !18
  %dev_id = load i32, i32* %9, align 4, !dbg !18
  %10 = getelementptr inbounds %1, %1* %arg.T_layout_trans47, i64 0, i32 0, !dbg !18
  %T_layout_trans = load i8*, i8** %10, align 8, !dbg !18
  %ptrint1 = ptrtoint i8* %T_layout_trans to i64, !dbg !18
  %maskedptr2 = and i64 %ptrint1, 63, !dbg !18
  %maskcond3 = icmp eq i64 %maskedptr2, 0, !dbg !18
  tail call void @llvm.assume(i1 %maskcond3), !dbg !18
  %11 = getelementptr inbounds %1, %1* %arg.T_layout_trans47, i64 0, i32 4, !dbg !18
  %arg.T_layout_trans.shape = load i64*, i64** %11, align 8, !dbg !18
  %12 = getelementptr inbounds %1, %1* %arg.T_layout_trans47, i64 0, i32 5, !dbg !18
  %arg.T_layout_trans.strides = load i64*, i64** %12, align 8, !dbg !18
  switch i32 %arg.p0.code, label %assert_fail4 [
    i32 13, label %assert_end5
    i32 7, label %assert_end5
    i32 4, label %assert_end5
    i32 3, label %assert_end5
  ], !dbg !18

assert_fail4:                                     ; preds = %assert_end
  %13 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %13(i8* getelementptr inbounds ([159 x i8], [159 x i8]* @.str.1, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_end5:                                      ; preds = %assert_end, %assert_end, %assert_end, %assert_end
  switch i32 %arg.T_layout_trans.code, label %assert_fail6 [
    i32 13, label %assert_end7
    i32 7, label %assert_end7
    i32 4, label %assert_end7
    i32 3, label %assert_end7
  ], !dbg !18

assert_fail6:                                     ; preds = %assert_end5
  %14 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %14(i8* getelementptr inbounds ([207 x i8], [207 x i8]* @.str.2, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_end7:                                      ; preds = %assert_end5, %assert_end5, %assert_end5, %assert_end5
  %15 = getelementptr inbounds %1, %1* %arg.p046, i64 0, i32 2, !dbg !18
  %16 = load i32, i32* %15, align 4, !dbg !18
  %17 = icmp eq i32 %16, 2, !dbg !18
  br i1 %17, label %assert_end11, label %assert_fail8, !dbg !18, !prof !19

assert_fail8:                                     ; preds = %assert_end7
  %18 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %18(i8* getelementptr inbounds ([94 x i8], [94 x i8]* @.str.3, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_end11:                                     ; preds = %assert_end7
  %19 = getelementptr inbounds %1, %1* %arg.p046, i64 0, i32 3, i32 2, !dbg !18
  %20 = load i16, i16* %19, align 2, !dbg !18
  %21 = icmp eq i16 %20, 1, !dbg !18
  %22 = getelementptr inbounds %1, %1* %arg.p046, i64 0, i32 3, i32 1, !dbg !18
  %23 = load i8, i8* %22, align 1, !dbg !18
  %24 = icmp eq i8 %23, 32, !dbg !18
  %25 = getelementptr inbounds %1, %1* %arg.p046, i64 0, i32 3, i32 0, !dbg !18
  %26 = load i8, i8* %25, align 1, !dbg !18
  %27 = icmp eq i8 %26, 2, !dbg !18
  %28 = and i1 %24, %27, !dbg !18
  %29 = and i1 %21, %28, !dbg !18
  br i1 %29, label %assert_end13, label %assert_fail12, !dbg !18, !prof !19

assert_fail12:                                    ; preds = %assert_end11
  %30 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %30(i8* getelementptr inbounds ([226 x i8], [226 x i8]* @.str.4, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_end13:                                     ; preds = %assert_end11
  %31 = load i64, i64* %arg.p0.shape, align 8, !dbg !18, !tbaa !36
  %32 = trunc i64 %31 to i32, !dbg !18
  %33 = icmp eq i32 %32, 512, !dbg !18
  br i1 %33, label %assert_end15, label %assert_fail14, !dbg !18, !prof !19

assert_fail14:                                    ; preds = %assert_end13
  %34 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %34(i8* getelementptr inbounds ([150 x i8], [150 x i8]* @.str.5, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_end15:                                     ; preds = %assert_end13
  %35 = getelementptr inbounds i64, i64* %arg.p0.shape, i64 1, !dbg !18
  %36 = load i64, i64* %35, align 8, !dbg !18, !tbaa !46
  %37 = trunc i64 %36 to i32, !dbg !18
  %38 = icmp eq i32 %37, 512, !dbg !18
  br i1 %38, label %assert_end17, label %assert_fail16, !dbg !18, !prof !19

assert_fail16:                                    ; preds = %assert_end15
  %39 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %39(i8* getelementptr inbounds ([150 x i8], [150 x i8]* @.str.6, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_end17:                                     ; preds = %assert_end15
  %40 = icmp eq i64* %arg.p0.strides, null, !dbg !18
  br i1 %40, label %if_end, label %if_then, !dbg !18, !prof !48

if_then:                                          ; preds = %assert_end17
  %41 = load i64, i64* %arg.p0.strides, align 8, !dbg !18, !tbaa !49
  %42 = trunc i64 %41 to i32, !dbg !18
  %43 = icmp eq i32 %42, 512, !dbg !18
  %44 = getelementptr inbounds i64, i64* %arg.p0.strides, i64 1, !dbg !18
  %45 = load i64, i64* %44, align 8, !dbg !18, !tbaa !59
  %46 = trunc i64 %45 to i32, !dbg !18
  %47 = icmp eq i32 %46, 1, !dbg !18
  %48 = and i1 %43, %47, !dbg !18
  br i1 %48, label %if_end, label %assert_fail18, !dbg !18, !prof !19

if_end:                                           ; preds = %assert_end17, %if_then
  %49 = getelementptr inbounds %1, %1* %arg.p046, i64 0, i32 6, !dbg !18
  %50 = load i64, i64* %49, align 8, !dbg !18
  %51 = icmp eq i64 %50, 0, !dbg !18
  br i1 %51, label %assert_end21, label %assert_fail20, !dbg !18, !prof !19

assert_fail18:                                    ; preds = %if_then
  %52 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %52(i8* getelementptr inbounds ([145 x i8], [145 x i8]* @.str.7, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_fail20:                                    ; preds = %if_end
  %53 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %53(i8* getelementptr inbounds ([185 x i8], [185 x i8]* @.str.8, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_end21:                                     ; preds = %if_end
  %54 = getelementptr inbounds %1, %1* %arg.p046, i64 0, i32 1, i32 0, !dbg !18
  %55 = load i32, i32* %54, align 4, !dbg !18
  %56 = icmp eq i32 %55, 1, !dbg !18
  br i1 %56, label %assert_end23, label %assert_fail22, !dbg !18, !prof !19

assert_fail22:                                    ; preds = %assert_end21
  %57 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %57(i8* getelementptr inbounds ([165 x i8], [165 x i8]* @.str.9, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_end23:                                     ; preds = %assert_end21
  %58 = getelementptr inbounds %1, %1* %arg.T_layout_trans47, i64 0, i32 2, !dbg !18
  %59 = load i32, i32* %58, align 4, !dbg !18
  %60 = icmp eq i32 %59, 3, !dbg !18
  br i1 %60, label %assert_end27, label %assert_fail24, !dbg !18, !prof !19

assert_fail24:                                    ; preds = %assert_end23
  %61 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %61(i8* getelementptr inbounds ([118 x i8], [118 x i8]* @.str.10, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_end27:                                     ; preds = %assert_end23
  %62 = getelementptr inbounds %1, %1* %arg.T_layout_trans47, i64 0, i32 3, i32 2, !dbg !18
  %63 = load i16, i16* %62, align 2, !dbg !18
  %64 = icmp eq i16 %63, 1, !dbg !18
  %65 = getelementptr inbounds %1, %1* %arg.T_layout_trans47, i64 0, i32 3, i32 1, !dbg !18
  %66 = load i8, i8* %65, align 1, !dbg !18
  %67 = icmp eq i8 %66, 32, !dbg !18
  %68 = getelementptr inbounds %1, %1* %arg.T_layout_trans47, i64 0, i32 3, i32 0, !dbg !18
  %69 = load i8, i8* %68, align 1, !dbg !18
  %70 = icmp eq i8 %69, 2, !dbg !18
  %71 = and i1 %67, %70, !dbg !18
  %72 = and i1 %64, %71, !dbg !18
  br i1 %72, label %assert_end29, label %assert_fail28, !dbg !18, !prof !19

assert_fail28:                                    ; preds = %assert_end27
  %73 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %73(i8* getelementptr inbounds ([274 x i8], [274 x i8]* @.str.11, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_end29:                                     ; preds = %assert_end27
  %74 = load i64, i64* %arg.T_layout_trans.shape, align 8, !dbg !18, !tbaa !61
  %75 = trunc i64 %74 to i32, !dbg !18
  %76 = icmp eq i32 %75, 32, !dbg !18
  br i1 %76, label %assert_end31, label %assert_fail30, !dbg !18, !prof !19

assert_fail30:                                    ; preds = %assert_end29
  %77 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %77(i8* getelementptr inbounds ([184 x i8], [184 x i8]* @.str.12, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_end31:                                     ; preds = %assert_end29
  %78 = getelementptr inbounds i64, i64* %arg.T_layout_trans.shape, i64 1, !dbg !18
  %79 = load i64, i64* %78, align 8, !dbg !18, !tbaa !71
  %80 = trunc i64 %79 to i32, !dbg !18
  %81 = icmp eq i32 %80, 512, !dbg !18
  br i1 %81, label %assert_end33, label %assert_fail32, !dbg !18, !prof !19

assert_fail32:                                    ; preds = %assert_end31
  %82 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %82(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.13, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_end33:                                     ; preds = %assert_end31
  %83 = getelementptr inbounds i64, i64* %arg.T_layout_trans.shape, i64 2, !dbg !18
  %84 = load i64, i64* %83, align 8, !dbg !18, !tbaa !73
  %85 = trunc i64 %84 to i32, !dbg !18
  %86 = icmp eq i32 %85, 16, !dbg !18
  br i1 %86, label %assert_end35, label %assert_fail34, !dbg !18, !prof !19

assert_fail34:                                    ; preds = %assert_end33
  %87 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %87(i8* getelementptr inbounds ([184 x i8], [184 x i8]* @.str.14, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_end35:                                     ; preds = %assert_end33
  %88 = icmp eq i64* %arg.T_layout_trans.strides, null, !dbg !18
  br i1 %88, label %if_end37, label %if_then36, !dbg !18, !prof !48

if_then36:                                        ; preds = %assert_end35
  %89 = load i64, i64* %arg.T_layout_trans.strides, align 8, !dbg !18, !tbaa !76
  %90 = trunc i64 %89 to i32, !dbg !18
  %91 = icmp eq i32 %90, 8192, !dbg !18
  %92 = getelementptr inbounds i64, i64* %arg.T_layout_trans.strides, i64 1, !dbg !18
  %93 = load i64, i64* %92, align 8, !dbg !18, !tbaa !86
  %94 = trunc i64 %93 to i32, !dbg !18
  %95 = icmp eq i32 %94, 16, !dbg !18
  %96 = getelementptr inbounds i64, i64* %arg.T_layout_trans.strides, i64 2, !dbg !18
  %97 = load i64, i64* %96, align 8, !dbg !18, !tbaa !88
  %98 = trunc i64 %97 to i32, !dbg !18
  %99 = icmp eq i32 %98, 1, !dbg !18
  %100 = and i1 %95, %99, !dbg !18
  %101 = and i1 %91, %100, !dbg !18
  br i1 %101, label %if_end37, label %assert_fail38, !dbg !18, !prof !19

if_end37:                                         ; preds = %assert_end35, %if_then36
  %102 = getelementptr inbounds %1, %1* %arg.T_layout_trans47, i64 0, i32 6, !dbg !18
  %103 = load i64, i64* %102, align 8, !dbg !18
  %104 = icmp eq i64 %103, 0, !dbg !18
  br i1 %104, label %assert_end41, label %assert_fail40, !dbg !18, !prof !19

assert_fail38:                                    ; preds = %if_then36
  %105 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %105(i8* getelementptr inbounds ([239 x i8], [239 x i8]* @.str.15, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_fail40:                                    ; preds = %if_end37
  %106 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %106(i8* getelementptr inbounds ([221 x i8], [221 x i8]* @.str.16, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_end41:                                     ; preds = %if_end37
  %107 = getelementptr inbounds %1, %1* %arg.T_layout_trans47, i64 0, i32 1, i32 0, !dbg !18
  %108 = load i32, i32* %107, align 4, !dbg !18
  %109 = icmp eq i32 %108, 1, !dbg !18
  br i1 %109, label %assert_end43, label %assert_fail42, !dbg !18, !prof !19

assert_fail42:                                    ; preds = %assert_end41
  %110 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %110(i8* getelementptr inbounds ([201 x i8], [201 x i8]* @.str.17, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_end43:                                     ; preds = %assert_end41
  %111 = getelementptr inbounds %1, %1* %arg.T_layout_trans47, i64 0, i32 1, i32 1, !dbg !18
  %112 = load i32, i32* %111, align 4, !dbg !18
  %113 = icmp eq i32 %dev_id, %112, !dbg !18
  br i1 %113, label %assert_end45, label %assert_fail44, !dbg !18, !prof !19

assert_fail44:                                    ; preds = %assert_end43
  %114 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !18, !tbaa !20
  tail call void %114(i8* getelementptr inbounds ([207 x i8], [207 x i8]* @.str.18, i64 0, i64 0)), !dbg !18
  ret i32 -1, !dbg !18

assert_end45:                                     ; preds = %assert_end43
  %115 = tail call fastcc i32 @tvmgen_default_fused_layout_transform_compute_(i8* %T_layout_trans, i8* %p0), !dbg !18
  ret i32 %115, !dbg !18
}

; Function Attrs: nounwind willreturn
declare void @llvm.assume(i1) #1

; Function Attrs: noinline
define internal fastcc i32 @tvmgen_default_fused_layout_transform_compute_(i8* noalias align 64 %0, i8* noalias align 64 %1) unnamed_addr #2 {
entry:
  %2 = alloca %closure_loop_parallel_ax0.ax1.fused, align 8
  %3 = getelementptr inbounds %closure_loop_parallel_ax0.ax1.fused, %closure_loop_parallel_ax0.ax1.fused* %2, i64 0, i32 0
  store i8* %0, i8** %3, align 8
  %4 = getelementptr inbounds %closure_loop_parallel_ax0.ax1.fused, %closure_loop_parallel_ax0.ax1.fused* %2, i64 0, i32 1
  store i8* %1, i8** %4, align 8
  %5 = load i32 (i32 (i32, %0*, i8*)*, i8*, i32)*, i32 (i32 (i32, %0*, i8*)*, i8*, i32)** @__TVMBackendParallelLaunch, align 8, !tbaa !20
  %6 = bitcast %closure_loop_parallel_ax0.ax1.fused* %2 to i8*
  %7 = call i32 %5(i32 (i32, %0*, i8*)* nonnull @__tvm_parallel_lambda, i8* nonnull %6, i32 0)
  ret i32 %7
}

; Function Attrs: nofree norecurse nounwind
define private i32 @__tvm_parallel_lambda(i32 %task_id, %0* nocapture readonly %0, i8* nocapture readonly %1) #3 {
parallel_closure_entry:
  %2 = bitcast i8* %1 to float**
  %T_layout_trans1 = load float*, float** %2, align 8
  %3 = getelementptr inbounds i8, i8* %1, i64 8
  %4 = bitcast i8* %3 to float**
  %p02 = load float*, float** %4, align 8
  %5 = getelementptr inbounds %0, %0* %0, i64 0, i32 1
  %num_task = load i32, i32* %5, align 4
  %6 = add nsw i32 %num_task, 16383
  %7 = sdiv i32 %6, %num_task
  %8 = add nsw i32 %task_id, 1
  %9 = mul nsw i32 %7, %8
  %10 = icmp slt i32 %9, 16384
  %11 = select i1 %10, i32 %9, i32 16384
  %12 = mul nsw i32 %7, %task_id
  %13 = icmp slt i32 %12, 16384
  %14 = select i1 %13, i32 %12, i32 16384
  %15 = icmp slt i32 %14, %11
  br i1 %15, label %for_body_ax0.ax1.fused.preheader, label %for_end_ax0.ax1.fused, !prof !19

for_body_ax0.ax1.fused.preheader:                 ; preds = %parallel_closure_entry
  %16 = sext i32 %14 to i64
  %17 = sext i32 %11 to i64
  br label %for_body_ax0.ax1.fused

for_body_ax0.ax1.fused:                           ; preds = %for_body_ax0.ax1.fused.preheader, %for_body_ax0.ax1.fused
  %indvars.iv = phi i64 [ %16, %for_body_ax0.ax1.fused.preheader ], [ %indvars.iv.next, %for_body_ax0.ax1.fused ]
  %ax0.ax1.fused3 = phi i32 [ %14, %for_body_ax0.ax1.fused.preheader ], [ %106, %for_body_ax0.ax1.fused ]
  %18 = trunc i64 %indvars.iv to i32
  %19 = and i32 %18, 511
  %20 = ashr i32 %18, 9
  %21 = shl nsw i32 %20, 13
  %22 = or i32 %21, %19
  %23 = sext i32 %22 to i64
  %24 = getelementptr inbounds float, float* %p02, i64 %23
  %25 = load float, float* %24, align 4, !tbaa !91
  %26 = or i32 %22, 512
  %27 = sext i32 %26 to i64
  %28 = getelementptr inbounds float, float* %p02, i64 %27
  %29 = load float, float* %28, align 4, !tbaa !91
  %30 = or i32 %22, 1024
  %31 = sext i32 %30 to i64
  %32 = getelementptr inbounds float, float* %p02, i64 %31
  %33 = load float, float* %32, align 4, !tbaa !91
  %34 = or i32 %22, 1536
  %35 = sext i32 %34 to i64
  %36 = getelementptr inbounds float, float* %p02, i64 %35
  %37 = load float, float* %36, align 4, !tbaa !91
  %38 = or i32 %22, 2048
  %39 = sext i32 %38 to i64
  %40 = getelementptr inbounds float, float* %p02, i64 %39
  %41 = load float, float* %40, align 4, !tbaa !91
  %42 = or i32 %22, 2560
  %43 = sext i32 %42 to i64
  %44 = getelementptr inbounds float, float* %p02, i64 %43
  %45 = load float, float* %44, align 4, !tbaa !91
  %46 = or i32 %22, 3072
  %47 = sext i32 %46 to i64
  %48 = getelementptr inbounds float, float* %p02, i64 %47
  %49 = load float, float* %48, align 4, !tbaa !91
  %50 = or i32 %22, 3584
  %51 = sext i32 %50 to i64
  %52 = getelementptr inbounds float, float* %p02, i64 %51
  %53 = load float, float* %52, align 4, !tbaa !91
  %54 = or i32 %22, 4096
  %55 = sext i32 %54 to i64
  %56 = getelementptr inbounds float, float* %p02, i64 %55
  %57 = load float, float* %56, align 4, !tbaa !91
  %58 = or i32 %22, 4608
  %59 = sext i32 %58 to i64
  %60 = getelementptr inbounds float, float* %p02, i64 %59
  %61 = load float, float* %60, align 4, !tbaa !91
  %62 = or i32 %22, 5120
  %63 = sext i32 %62 to i64
  %64 = getelementptr inbounds float, float* %p02, i64 %63
  %65 = load float, float* %64, align 4, !tbaa !91
  %66 = or i32 %22, 5632
  %67 = sext i32 %66 to i64
  %68 = getelementptr inbounds float, float* %p02, i64 %67
  %69 = load float, float* %68, align 4, !tbaa !91
  %70 = or i32 %22, 6144
  %71 = sext i32 %70 to i64
  %72 = getelementptr inbounds float, float* %p02, i64 %71
  %73 = load float, float* %72, align 4, !tbaa !91
  %74 = or i32 %22, 6656
  %75 = sext i32 %74 to i64
  %76 = getelementptr inbounds float, float* %p02, i64 %75
  %77 = load float, float* %76, align 4, !tbaa !91
  %78 = or i32 %22, 7168
  %79 = sext i32 %78 to i64
  %80 = getelementptr inbounds float, float* %p02, i64 %79
  %81 = load float, float* %80, align 4, !tbaa !91
  %82 = or i32 %22, 7680
  %83 = sext i32 %82 to i64
  %84 = getelementptr inbounds float, float* %p02, i64 %83
  %85 = load float, float* %84, align 4, !tbaa !91
  %86 = insertelement <16 x float> undef, float %25, i32 0
  %87 = insertelement <16 x float> %86, float %29, i32 1
  %88 = insertelement <16 x float> %87, float %33, i32 2
  %89 = insertelement <16 x float> %88, float %37, i32 3
  %90 = insertelement <16 x float> %89, float %41, i32 4
  %91 = insertelement <16 x float> %90, float %45, i32 5
  %92 = insertelement <16 x float> %91, float %49, i32 6
  %93 = insertelement <16 x float> %92, float %53, i32 7
  %94 = insertelement <16 x float> %93, float %57, i32 8
  %95 = insertelement <16 x float> %94, float %61, i32 9
  %96 = insertelement <16 x float> %95, float %65, i32 10
  %97 = insertelement <16 x float> %96, float %69, i32 11
  %98 = insertelement <16 x float> %97, float %73, i32 12
  %99 = insertelement <16 x float> %98, float %77, i32 13
  %100 = insertelement <16 x float> %99, float %81, i32 14
  %101 = insertelement <16 x float> %100, float %85, i32 15
  %102 = shl nsw i32 %ax0.ax1.fused3, 4
  %103 = sext i32 %102 to i64
  %104 = getelementptr inbounds float, float* %T_layout_trans1, i64 %103
  %105 = bitcast float* %104 to <16 x float>*
  store <16 x float> %101, <16 x float>* %105, align 64, !tbaa !93
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %106 = add nsw i32 %ax0.ax1.fused3, 1
  %107 = icmp slt i64 %indvars.iv.next, %17
  br i1 %107, label %for_body_ax0.ax1.fused, label %for_end_ax0.ax1.fused, !prof !19

for_end_ax0.ax1.fused:                            ; preds = %for_body_ax0.ax1.fused, %parallel_closure_entry
  ret i32 0
}

define dllexport i32 @tvmgen_default_fused_nn_contrib_dense_pack(i8* noalias nocapture readonly %args, i32* noalias nocapture readonly %arg_type_ids, i32 %num_args, i8* noalias nocapture readnone %out_ret_value, i32* noalias nocapture readnone %out_ret_tcode, i8* noalias nocapture readnone %resource_handle) local_unnamed_addr #0 !dbg !95 {
entry:
  call void @llvm.dbg.value(metadata i8* %args, metadata !97, metadata !DIExpression()), !dbg !103
  call void @llvm.dbg.value(metadata i32* %arg_type_ids, metadata !98, metadata !DIExpression()), !dbg !103
  call void @llvm.dbg.value(metadata i32 %num_args, metadata !99, metadata !DIExpression()), !dbg !103
  call void @llvm.dbg.value(metadata i8* %out_ret_value, metadata !100, metadata !DIExpression()), !dbg !103
  call void @llvm.dbg.value(metadata i32* %out_ret_tcode, metadata !101, metadata !DIExpression()), !dbg !103
  call void @llvm.dbg.value(metadata i8* %resource_handle, metadata !102, metadata !DIExpression()), !dbg !103
  %0 = icmp eq i32 %num_args, 3, !dbg !103
  br i1 %0, label %assert_end, label %assert_fail, !dbg !103, !prof !19

assert_fail:                                      ; preds = %entry
  %1 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %1(i8* getelementptr inbounds ([93 x i8], [93 x i8]* @.str.19, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end:                                       ; preds = %entry
  %2 = bitcast i8* %args to %1**, !dbg !103
  %arg.p071 = load %1*, %1** %2, align 8, !dbg !103
  %arg.p0.code = load i32, i32* %arg_type_ids, align 4, !dbg !103, !tbaa !104
  %3 = getelementptr inbounds i8, i8* %args, i64 8, !dbg !103
  %4 = bitcast i8* %3 to %1**, !dbg !103
  %arg.p172 = load %1*, %1** %4, align 8, !dbg !103
  %5 = getelementptr inbounds i32, i32* %arg_type_ids, i64 1, !dbg !103
  %arg.p1.code = load i32, i32* %5, align 4, !dbg !103, !tbaa !115
  %6 = getelementptr inbounds i8, i8* %args, i64 16, !dbg !103
  %7 = bitcast i8* %6 to %1**, !dbg !103
  %arg.compute73 = load %1*, %1** %7, align 8, !dbg !103
  %8 = getelementptr inbounds i32, i32* %arg_type_ids, i64 2, !dbg !103
  %arg.compute.code = load i32, i32* %8, align 4, !dbg !103, !tbaa !117
  %9 = getelementptr inbounds %1, %1* %arg.p071, i64 0, i32 0, !dbg !103
  %p0 = load i8*, i8** %9, align 8, !dbg !103
  %ptrint = ptrtoint i8* %p0 to i64, !dbg !103
  %maskedptr = and i64 %ptrint, 63, !dbg !103
  %maskcond = icmp eq i64 %maskedptr, 0, !dbg !103
  tail call void @llvm.assume(i1 %maskcond), !dbg !103
  %10 = getelementptr inbounds %1, %1* %arg.p071, i64 0, i32 4, !dbg !103
  %arg.p0.shape = load i64*, i64** %10, align 8, !dbg !103
  %11 = getelementptr inbounds %1, %1* %arg.p071, i64 0, i32 5, !dbg !103
  %arg.p0.strides = load i64*, i64** %11, align 8, !dbg !103
  %12 = getelementptr inbounds %1, %1* %arg.p071, i64 0, i32 1, i32 1, !dbg !103
  %dev_id = load i32, i32* %12, align 4, !dbg !103
  %13 = getelementptr inbounds %1, %1* %arg.p172, i64 0, i32 0, !dbg !103
  %p1 = load i8*, i8** %13, align 8, !dbg !103
  %ptrint1 = ptrtoint i8* %p1 to i64, !dbg !103
  %maskedptr2 = and i64 %ptrint1, 63, !dbg !103
  %maskcond3 = icmp eq i64 %maskedptr2, 0, !dbg !103
  tail call void @llvm.assume(i1 %maskcond3), !dbg !103
  %14 = getelementptr inbounds %1, %1* %arg.p172, i64 0, i32 4, !dbg !103
  %arg.p1.shape = load i64*, i64** %14, align 8, !dbg !103
  %15 = getelementptr inbounds %1, %1* %arg.p172, i64 0, i32 5, !dbg !103
  %arg.p1.strides = load i64*, i64** %15, align 8, !dbg !103
  %16 = getelementptr inbounds %1, %1* %arg.compute73, i64 0, i32 0, !dbg !103
  %compute = load i8*, i8** %16, align 8, !dbg !103
  %ptrint4 = ptrtoint i8* %compute to i64, !dbg !103
  %maskedptr5 = and i64 %ptrint4, 63, !dbg !103
  %maskcond6 = icmp eq i64 %maskedptr5, 0, !dbg !103
  tail call void @llvm.assume(i1 %maskcond6), !dbg !103
  %17 = getelementptr inbounds %1, %1* %arg.compute73, i64 0, i32 4, !dbg !103
  %arg.compute.shape = load i64*, i64** %17, align 8, !dbg !103
  %18 = getelementptr inbounds %1, %1* %arg.compute73, i64 0, i32 5, !dbg !103
  %arg.compute.strides = load i64*, i64** %18, align 8, !dbg !103
  switch i32 %arg.p0.code, label %assert_fail7 [
    i32 13, label %assert_end8
    i32 7, label %assert_end8
    i32 4, label %assert_end8
    i32 3, label %assert_end8
  ], !dbg !103

assert_fail7:                                     ; preds = %assert_end
  %19 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %19(i8* getelementptr inbounds ([164 x i8], [164 x i8]* @.str.20, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end8:                                      ; preds = %assert_end, %assert_end, %assert_end, %assert_end
  switch i32 %arg.p1.code, label %assert_fail9 [
    i32 13, label %assert_end10
    i32 7, label %assert_end10
    i32 4, label %assert_end10
    i32 3, label %assert_end10
  ], !dbg !103

assert_fail9:                                     ; preds = %assert_end8
  %20 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %20(i8* getelementptr inbounds ([164 x i8], [164 x i8]* @.str.21, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end10:                                     ; preds = %assert_end8, %assert_end8, %assert_end8, %assert_end8
  switch i32 %arg.compute.code, label %assert_fail11 [
    i32 13, label %assert_end12
    i32 7, label %assert_end12
    i32 4, label %assert_end12
    i32 3, label %assert_end12
  ], !dbg !103

assert_fail11:                                    ; preds = %assert_end10
  %21 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %21(i8* getelementptr inbounds ([184 x i8], [184 x i8]* @.str.22, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end12:                                     ; preds = %assert_end10, %assert_end10, %assert_end10, %assert_end10
  %22 = getelementptr inbounds %1, %1* %arg.p071, i64 0, i32 2, !dbg !103
  %23 = load i32, i32* %22, align 4, !dbg !103
  %24 = icmp eq i32 %23, 2, !dbg !103
  br i1 %24, label %assert_end16, label %assert_fail13, !dbg !103, !prof !19

assert_fail13:                                    ; preds = %assert_end12
  %25 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %25(i8* getelementptr inbounds ([94 x i8], [94 x i8]* @.str.3, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end16:                                     ; preds = %assert_end12
  %26 = getelementptr inbounds %1, %1* %arg.p071, i64 0, i32 3, i32 2, !dbg !103
  %27 = load i16, i16* %26, align 2, !dbg !103
  %28 = icmp eq i16 %27, 1, !dbg !103
  %29 = getelementptr inbounds %1, %1* %arg.p071, i64 0, i32 3, i32 1, !dbg !103
  %30 = load i8, i8* %29, align 1, !dbg !103
  %31 = icmp eq i8 %30, 32, !dbg !103
  %32 = getelementptr inbounds %1, %1* %arg.p071, i64 0, i32 3, i32 0, !dbg !103
  %33 = load i8, i8* %32, align 1, !dbg !103
  %34 = icmp eq i8 %33, 2, !dbg !103
  %35 = and i1 %31, %34, !dbg !103
  %36 = and i1 %28, %35, !dbg !103
  br i1 %36, label %assert_end18, label %assert_fail17, !dbg !103, !prof !19

assert_fail17:                                    ; preds = %assert_end16
  %37 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %37(i8* getelementptr inbounds ([226 x i8], [226 x i8]* @.str.4, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end18:                                     ; preds = %assert_end16
  %38 = load i64, i64* %arg.p0.shape, align 8, !dbg !103, !tbaa !120
  %39 = trunc i64 %38 to i32, !dbg !103
  %40 = icmp eq i32 %39, 512, !dbg !103
  br i1 %40, label %assert_end20, label %assert_fail19, !dbg !103, !prof !19

assert_fail19:                                    ; preds = %assert_end18
  %41 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %41(i8* getelementptr inbounds ([150 x i8], [150 x i8]* @.str.5, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end20:                                     ; preds = %assert_end18
  %42 = getelementptr inbounds i64, i64* %arg.p0.shape, i64 1, !dbg !103
  %43 = load i64, i64* %42, align 8, !dbg !103, !tbaa !130
  %44 = trunc i64 %43 to i32, !dbg !103
  %45 = icmp eq i32 %44, 512, !dbg !103
  br i1 %45, label %assert_end22, label %assert_fail21, !dbg !103, !prof !19

assert_fail21:                                    ; preds = %assert_end20
  %46 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %46(i8* getelementptr inbounds ([150 x i8], [150 x i8]* @.str.6, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end22:                                     ; preds = %assert_end20
  %47 = icmp eq i64* %arg.p0.strides, null, !dbg !103
  br i1 %47, label %if_end, label %if_then, !dbg !103, !prof !48

if_then:                                          ; preds = %assert_end22
  %48 = load i64, i64* %arg.p0.strides, align 8, !dbg !103, !tbaa !132
  %49 = trunc i64 %48 to i32, !dbg !103
  %50 = icmp eq i32 %49, 512, !dbg !103
  %51 = getelementptr inbounds i64, i64* %arg.p0.strides, i64 1, !dbg !103
  %52 = load i64, i64* %51, align 8, !dbg !103, !tbaa !142
  %53 = trunc i64 %52 to i32, !dbg !103
  %54 = icmp eq i32 %53, 1, !dbg !103
  %55 = and i1 %50, %54, !dbg !103
  br i1 %55, label %if_end, label %assert_fail23, !dbg !103, !prof !19

if_end:                                           ; preds = %assert_end22, %if_then
  %56 = getelementptr inbounds %1, %1* %arg.p071, i64 0, i32 6, !dbg !103
  %57 = load i64, i64* %56, align 8, !dbg !103
  %58 = icmp eq i64 %57, 0, !dbg !103
  br i1 %58, label %assert_end26, label %assert_fail25, !dbg !103, !prof !19

assert_fail23:                                    ; preds = %if_then
  %59 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %59(i8* getelementptr inbounds ([145 x i8], [145 x i8]* @.str.7, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_fail25:                                    ; preds = %if_end
  %60 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %60(i8* getelementptr inbounds ([185 x i8], [185 x i8]* @.str.8, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end26:                                     ; preds = %if_end
  %61 = getelementptr inbounds %1, %1* %arg.p071, i64 0, i32 1, i32 0, !dbg !103
  %62 = load i32, i32* %61, align 4, !dbg !103
  %63 = icmp eq i32 %62, 1, !dbg !103
  br i1 %63, label %assert_end28, label %assert_fail27, !dbg !103, !prof !19

assert_fail27:                                    ; preds = %assert_end26
  %64 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %64(i8* getelementptr inbounds ([165 x i8], [165 x i8]* @.str.9, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end28:                                     ; preds = %assert_end26
  %65 = getelementptr inbounds %1, %1* %arg.p172, i64 0, i32 2, !dbg !103
  %66 = load i32, i32* %65, align 4, !dbg !103
  %67 = icmp eq i32 %66, 3, !dbg !103
  br i1 %67, label %assert_end32, label %assert_fail29, !dbg !103, !prof !19

assert_fail29:                                    ; preds = %assert_end28
  %68 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %68(i8* getelementptr inbounds ([94 x i8], [94 x i8]* @.str.23, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end32:                                     ; preds = %assert_end28
  %69 = getelementptr inbounds %1, %1* %arg.p172, i64 0, i32 3, i32 2, !dbg !103
  %70 = load i16, i16* %69, align 2, !dbg !103
  %71 = icmp eq i16 %70, 1, !dbg !103
  %72 = getelementptr inbounds %1, %1* %arg.p172, i64 0, i32 3, i32 1, !dbg !103
  %73 = load i8, i8* %72, align 1, !dbg !103
  %74 = icmp eq i8 %73, 32, !dbg !103
  %75 = getelementptr inbounds %1, %1* %arg.p172, i64 0, i32 3, i32 0, !dbg !103
  %76 = load i8, i8* %75, align 1, !dbg !103
  %77 = icmp eq i8 %76, 2, !dbg !103
  %78 = and i1 %74, %77, !dbg !103
  %79 = and i1 %71, %78, !dbg !103
  br i1 %79, label %assert_end34, label %assert_fail33, !dbg !103, !prof !19

assert_fail33:                                    ; preds = %assert_end32
  %80 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %80(i8* getelementptr inbounds ([226 x i8], [226 x i8]* @.str.24, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end34:                                     ; preds = %assert_end32
  %81 = load i64, i64* %arg.p1.shape, align 8, !dbg !103, !tbaa !144
  %82 = trunc i64 %81 to i32, !dbg !103
  %83 = icmp eq i32 %82, 32, !dbg !103
  br i1 %83, label %assert_end36, label %assert_fail35, !dbg !103, !prof !19

assert_fail35:                                    ; preds = %assert_end34
  %84 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %84(i8* getelementptr inbounds ([148 x i8], [148 x i8]* @.str.25, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end36:                                     ; preds = %assert_end34
  %85 = getelementptr inbounds i64, i64* %arg.p1.shape, i64 1, !dbg !103
  %86 = load i64, i64* %85, align 8, !dbg !103, !tbaa !154
  %87 = trunc i64 %86 to i32, !dbg !103
  %88 = icmp eq i32 %87, 512, !dbg !103
  br i1 %88, label %assert_end38, label %assert_fail37, !dbg !103, !prof !19

assert_fail37:                                    ; preds = %assert_end36
  %89 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %89(i8* getelementptr inbounds ([150 x i8], [150 x i8]* @.str.26, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end38:                                     ; preds = %assert_end36
  %90 = getelementptr inbounds i64, i64* %arg.p1.shape, i64 2, !dbg !103
  %91 = load i64, i64* %90, align 8, !dbg !103, !tbaa !156
  %92 = trunc i64 %91 to i32, !dbg !103
  %93 = icmp eq i32 %92, 16, !dbg !103
  br i1 %93, label %assert_end40, label %assert_fail39, !dbg !103, !prof !19

assert_fail39:                                    ; preds = %assert_end38
  %94 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %94(i8* getelementptr inbounds ([148 x i8], [148 x i8]* @.str.27, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end40:                                     ; preds = %assert_end38
  %95 = icmp eq i64* %arg.p1.strides, null, !dbg !103
  br i1 %95, label %if_end42, label %if_then41, !dbg !103, !prof !48

if_then41:                                        ; preds = %assert_end40
  %96 = load i64, i64* %arg.p1.strides, align 8, !dbg !103, !tbaa !159
  %97 = trunc i64 %96 to i32, !dbg !103
  %98 = icmp eq i32 %97, 8192, !dbg !103
  %99 = getelementptr inbounds i64, i64* %arg.p1.strides, i64 1, !dbg !103
  %100 = load i64, i64* %99, align 8, !dbg !103, !tbaa !169
  %101 = trunc i64 %100 to i32, !dbg !103
  %102 = icmp eq i32 %101, 16, !dbg !103
  %103 = getelementptr inbounds i64, i64* %arg.p1.strides, i64 2, !dbg !103
  %104 = load i64, i64* %103, align 8, !dbg !103, !tbaa !171
  %105 = trunc i64 %104 to i32, !dbg !103
  %106 = icmp eq i32 %105, 1, !dbg !103
  %107 = and i1 %102, %106, !dbg !103
  %108 = and i1 %98, %107, !dbg !103
  br i1 %108, label %if_end42, label %assert_fail43, !dbg !103, !prof !19

if_end42:                                         ; preds = %assert_end40, %if_then41
  %109 = getelementptr inbounds %1, %1* %arg.p172, i64 0, i32 6, !dbg !103
  %110 = load i64, i64* %109, align 8, !dbg !103
  %111 = icmp eq i64 %110, 0, !dbg !103
  br i1 %111, label %assert_end46, label %assert_fail45, !dbg !103, !prof !19

assert_fail43:                                    ; preds = %if_then41
  %112 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %112(i8* getelementptr inbounds ([191 x i8], [191 x i8]* @.str.28, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_fail45:                                    ; preds = %if_end42
  %113 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %113(i8* getelementptr inbounds ([185 x i8], [185 x i8]* @.str.29, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end46:                                     ; preds = %if_end42
  %114 = getelementptr inbounds %1, %1* %arg.p172, i64 0, i32 1, i32 0, !dbg !103
  %115 = load i32, i32* %114, align 4, !dbg !103
  %116 = icmp eq i32 %115, 1, !dbg !103
  br i1 %116, label %assert_end48, label %assert_fail47, !dbg !103, !prof !19

assert_fail47:                                    ; preds = %assert_end46
  %117 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %117(i8* getelementptr inbounds ([165 x i8], [165 x i8]* @.str.30, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end48:                                     ; preds = %assert_end46
  %118 = getelementptr inbounds %1, %1* %arg.p172, i64 0, i32 1, i32 1, !dbg !103
  %119 = load i32, i32* %118, align 4, !dbg !103
  %120 = icmp eq i32 %dev_id, %119, !dbg !103
  br i1 %120, label %assert_end50, label %assert_fail49, !dbg !103, !prof !19

assert_fail49:                                    ; preds = %assert_end48
  %121 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %121(i8* getelementptr inbounds ([171 x i8], [171 x i8]* @.str.31, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end50:                                     ; preds = %assert_end48
  %122 = getelementptr inbounds %1, %1* %arg.compute73, i64 0, i32 2, !dbg !103
  %123 = load i32, i32* %122, align 4, !dbg !103
  %124 = icmp eq i32 %123, 2, !dbg !103
  br i1 %124, label %assert_end54, label %assert_fail51, !dbg !103, !prof !19

assert_fail51:                                    ; preds = %assert_end50
  %125 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %125(i8* getelementptr inbounds ([104 x i8], [104 x i8]* @.str.32, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end54:                                     ; preds = %assert_end50
  %126 = getelementptr inbounds %1, %1* %arg.compute73, i64 0, i32 3, i32 2, !dbg !103
  %127 = load i16, i16* %126, align 2, !dbg !103
  %128 = icmp eq i16 %127, 1, !dbg !103
  %129 = getelementptr inbounds %1, %1* %arg.compute73, i64 0, i32 3, i32 1, !dbg !103
  %130 = load i8, i8* %129, align 1, !dbg !103
  %131 = icmp eq i8 %130, 32, !dbg !103
  %132 = getelementptr inbounds %1, %1* %arg.compute73, i64 0, i32 3, i32 0, !dbg !103
  %133 = load i8, i8* %132, align 1, !dbg !103
  %134 = icmp eq i8 %133, 2, !dbg !103
  %135 = and i1 %131, %134, !dbg !103
  %136 = and i1 %128, %135, !dbg !103
  br i1 %136, label %assert_end56, label %assert_fail55, !dbg !103, !prof !19

assert_fail55:                                    ; preds = %assert_end54
  %137 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %137(i8* getelementptr inbounds ([246 x i8], [246 x i8]* @.str.33, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end56:                                     ; preds = %assert_end54
  %138 = load i64, i64* %arg.compute.shape, align 8, !dbg !103, !tbaa !174
  %139 = trunc i64 %138 to i32, !dbg !103
  %140 = icmp eq i32 %139, 512, !dbg !103
  br i1 %140, label %assert_end58, label %assert_fail57, !dbg !103, !prof !19

assert_fail57:                                    ; preds = %assert_end56
  %141 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %141(i8* getelementptr inbounds ([165 x i8], [165 x i8]* @.str.34, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end58:                                     ; preds = %assert_end56
  %142 = getelementptr inbounds i64, i64* %arg.compute.shape, i64 1, !dbg !103
  %143 = load i64, i64* %142, align 8, !dbg !103, !tbaa !184
  %144 = trunc i64 %143 to i32, !dbg !103
  %145 = icmp eq i32 %144, 512, !dbg !103
  br i1 %145, label %assert_end60, label %assert_fail59, !dbg !103, !prof !19

assert_fail59:                                    ; preds = %assert_end58
  %146 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %146(i8* getelementptr inbounds ([165 x i8], [165 x i8]* @.str.35, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end60:                                     ; preds = %assert_end58
  %147 = icmp eq i64* %arg.compute.strides, null, !dbg !103
  br i1 %147, label %if_end62, label %if_then61, !dbg !103, !prof !48

if_then61:                                        ; preds = %assert_end60
  %148 = load i64, i64* %arg.compute.strides, align 8, !dbg !103, !tbaa !186
  %149 = trunc i64 %148 to i32, !dbg !103
  %150 = icmp eq i32 %149, 512, !dbg !103
  %151 = getelementptr inbounds i64, i64* %arg.compute.strides, i64 1, !dbg !103
  %152 = load i64, i64* %151, align 8, !dbg !103, !tbaa !196
  %153 = trunc i64 %152 to i32, !dbg !103
  %154 = icmp eq i32 %153, 1, !dbg !103
  %155 = and i1 %150, %154, !dbg !103
  br i1 %155, label %if_end62, label %assert_fail63, !dbg !103, !prof !19

if_end62:                                         ; preds = %assert_end60, %if_then61
  %156 = getelementptr inbounds %1, %1* %arg.compute73, i64 0, i32 6, !dbg !103
  %157 = load i64, i64* %156, align 8, !dbg !103
  %158 = icmp eq i64 %157, 0, !dbg !103
  br i1 %158, label %assert_end66, label %assert_fail65, !dbg !103, !prof !19

assert_fail63:                                    ; preds = %if_then61
  %159 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %159(i8* getelementptr inbounds ([160 x i8], [160 x i8]* @.str.36, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_fail65:                                    ; preds = %if_end62
  %160 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %160(i8* getelementptr inbounds ([200 x i8], [200 x i8]* @.str.37, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end66:                                     ; preds = %if_end62
  %161 = getelementptr inbounds %1, %1* %arg.compute73, i64 0, i32 1, i32 0, !dbg !103
  %162 = load i32, i32* %161, align 4, !dbg !103
  %163 = icmp eq i32 %162, 1, !dbg !103
  br i1 %163, label %assert_end68, label %assert_fail67, !dbg !103, !prof !19

assert_fail67:                                    ; preds = %assert_end66
  %164 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %164(i8* getelementptr inbounds ([180 x i8], [180 x i8]* @.str.38, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end68:                                     ; preds = %assert_end66
  %165 = getelementptr inbounds %1, %1* %arg.compute73, i64 0, i32 1, i32 1, !dbg !103
  %166 = load i32, i32* %165, align 4, !dbg !103
  %167 = icmp eq i32 %dev_id, %166, !dbg !103
  br i1 %167, label %assert_end70, label %assert_fail69, !dbg !103, !prof !19

assert_fail69:                                    ; preds = %assert_end68
  %168 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 8, !dbg !103, !tbaa !20
  tail call void %168(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.39, i64 0, i64 0)), !dbg !103
  ret i32 -1, !dbg !103

assert_end70:                                     ; preds = %assert_end68
  %169 = tail call fastcc i32 @tvmgen_default_fused_nn_contrib_dense_pack_compute_(i8* %p0, i8* %p1, i8* %compute), !dbg !103
  ret i32 %169, !dbg !103
}

; Function Attrs: noinline
define internal fastcc i32 @tvmgen_default_fused_nn_contrib_dense_pack_compute_(i8* noalias align 64 %0, i8* noalias align 64 %1, i8* noalias align 64 %2) unnamed_addr #2 {
entry:
  %3 = alloca %closure_loop_parallel_x.outer.y.outer.fused, align 8
  %4 = getelementptr inbounds %closure_loop_parallel_x.outer.y.outer.fused, %closure_loop_parallel_x.outer.y.outer.fused* %3, i64 0, i32 0
  store i8* %0, i8** %4, align 8
  %5 = getelementptr inbounds %closure_loop_parallel_x.outer.y.outer.fused, %closure_loop_parallel_x.outer.y.outer.fused* %3, i64 0, i32 1
  store i8* %1, i8** %5, align 8
  %6 = getelementptr inbounds %closure_loop_parallel_x.outer.y.outer.fused, %closure_loop_parallel_x.outer.y.outer.fused* %3, i64 0, i32 2
  store i8* %2, i8** %6, align 8
  %7 = load i32 (i32 (i32, %0*, i8*)*, i8*, i32)*, i32 (i32 (i32, %0*, i8*)*, i8*, i32)** @__TVMBackendParallelLaunch, align 8, !tbaa !20
  %8 = bitcast %closure_loop_parallel_x.outer.y.outer.fused* %3 to i8*
  %9 = call i32 %7(i32 (i32, %0*, i8*)* nonnull @__tvm_parallel_lambda.40, i8* nonnull %8, i32 0)
  ret i32 %9
}

; Function Attrs: nofree nounwind
define private i32 @__tvm_parallel_lambda.40(i32 %task_id, %0* nocapture readonly %0, i8* nocapture readonly %1) #4 {
parallel_closure_entry:
  %2 = bitcast i8* %1 to float**
  %p02 = load float*, float** %2, align 8
  %3 = getelementptr inbounds i8, i8* %1, i64 8
  %4 = bitcast i8* %3 to float**
  %p13 = load float*, float** %4, align 8
  %5 = getelementptr inbounds i8, i8* %1, i64 16
  %6 = bitcast i8* %5 to float**
  %compute4 = load float*, float** %6, align 8
  %7 = getelementptr inbounds %0, %0* %0, i64 0, i32 1
  %num_task = load i32, i32* %7, align 4
  %8 = add nsw i32 %num_task, 31
  %9 = sdiv i32 %8, %num_task
  %10 = add nsw i32 %task_id, 1
  %11 = mul nsw i32 %9, %10
  %12 = icmp slt i32 %11, 32
  %13 = select i1 %12, i32 %11, i32 32
  %14 = mul nsw i32 %9, %task_id
  %15 = icmp slt i32 %14, 32
  %16 = select i1 %15, i32 %14, i32 32
  %17 = icmp slt i32 %16, %13
  br i1 %17, label %for_begin_y.inner.outer.x.inner.outer.fused.preheader, label %for_end_x.outer.y.outer.fused, !prof !19

for_begin_y.inner.outer.x.inner.outer.fused.preheader: ; preds = %parallel_closure_entry, %for_end_y.inner.outer.x.inner.outer.fused
  %x.outer.y.outer.fused75 = phi i32 [ %32, %for_end_y.inner.outer.x.inner.outer.fused ], [ %16, %parallel_closure_entry ]
  %18 = ashr i32 %x.outer.y.outer.fused75, 1
  %19 = shl i32 %18, 5
  %20 = shl i32 %x.outer.y.outer.fused75, 17
  %21 = and i32 %20, 131072
  %22 = shl nsw i32 %18, 14
  br label %for_body_y.inner.outer.x.inner.outer.fused

for_end_x.outer.y.outer.fused:                    ; preds = %for_end_y.inner.outer.x.inner.outer.fused, %parallel_closure_entry
  ret i32 0

for_body_y.inner.outer.x.inner.outer.fused:       ; preds = %for_end_k.outer, %for_begin_y.inner.outer.x.inner.outer.fused.preheader
  %y.inner.outer.x.inner.outer.fused74 = phi i32 [ 0, %for_begin_y.inner.outer.x.inner.outer.fused.preheader ], [ %500, %for_end_k.outer ]
  %23 = and i32 %y.inner.outer.x.inner.outer.fused74, 1
  %24 = lshr i32 %y.inner.outer.x.inner.outer.fused74, 1
  %25 = shl nuw nsw i32 %24, 11
  %26 = add nuw nsw i32 %25, %21
  %27 = add nsw i32 %26, %19
  %28 = shl nuw nsw i32 %23, 13
  %29 = or i32 %28, %22
  %30 = sext i32 %29 to i64
  %31 = zext i32 %26 to i64
  br label %for_body_k.outer

for_end_y.inner.outer.x.inner.outer.fused:        ; preds = %for_end_k.outer
  %32 = add nsw i32 %x.outer.y.outer.fused75, 1
  %33 = icmp slt i32 %32, %13
  br i1 %33, label %for_begin_y.inner.outer.x.inner.outer.fused.preheader, label %for_end_x.outer.y.outer.fused, !prof !19

for_body_k.outer:                                 ; preds = %for_body_k.outer, %for_body_y.inner.outer.x.inner.outer.fused
  %indvars.iv = phi i64 [ 0, %for_body_y.inner.outer.x.inner.outer.fused ], [ %indvars.iv.next, %for_body_k.outer ]
  %compute.global1.sroa.0.072 = phi <16 x float> [ zeroinitializer, %for_body_y.inner.outer.x.inner.outer.fused ], [ %465, %for_body_k.outer ]
  %compute.global1.sroa.34.071 = phi <16 x float> [ zeroinitializer, %for_body_y.inner.outer.x.inner.outer.fused ], [ %471, %for_body_k.outer ]
  %compute.global1.sroa.68.070 = phi <16 x float> [ zeroinitializer, %for_body_y.inner.outer.x.inner.outer.fused ], [ %477, %for_body_k.outer ]
  %compute.global1.sroa.102.069 = phi <16 x float> [ zeroinitializer, %for_body_y.inner.outer.x.inner.outer.fused ], [ %483, %for_body_k.outer ]
  %34 = shl nuw nsw i64 %indvars.iv, 4
  %35 = add nuw nsw i64 %34, %31
  %36 = shl nuw nsw i64 %indvars.iv, 8
  %37 = add nuw nsw i64 %36, %30
  %38 = or i64 %37, 96
  %39 = or i64 %37, 80
  %40 = or i64 %37, 64
  %41 = or i64 %37, 48
  %42 = or i64 %37, 32
  %43 = or i64 %37, 240
  %44 = or i64 %37, 224
  %45 = or i64 %37, 208
  %46 = or i64 %37, 192
  %47 = or i64 %37, 176
  %48 = or i64 %37, 160
  %49 = or i64 %37, 16
  %50 = or i64 %37, 144
  %51 = or i64 %37, 128
  %52 = or i64 %37, 112
  %53 = getelementptr inbounds float, float* %p02, i64 %35
  %54 = load float, float* %53, align 64, !tbaa !198
  %55 = insertelement <16 x float> undef, float %54, i32 0
  %56 = shufflevector <16 x float> %55, <16 x float> undef, <16 x i32> zeroinitializer
  %57 = getelementptr inbounds float, float* %p13, i64 %37
  %58 = bitcast float* %57 to <16 x float>*
  %59 = load <16 x float>, <16 x float>* %58, align 64, !tbaa !200
  %60 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %56, <16 x float> %59, <16 x float> %compute.global1.sroa.0.072)
  %61 = add nuw nsw i64 %35, 512
  %62 = getelementptr inbounds float, float* %p02, i64 %61
  %63 = load float, float* %62, align 64, !tbaa !198
  %64 = insertelement <16 x float> undef, float %63, i32 0
  %65 = shufflevector <16 x float> %64, <16 x float> undef, <16 x i32> zeroinitializer
  %66 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %65, <16 x float> %59, <16 x float> %compute.global1.sroa.34.071)
  %67 = add nuw nsw i64 %35, 1024
  %68 = getelementptr inbounds float, float* %p02, i64 %67
  %69 = load float, float* %68, align 64, !tbaa !198
  %70 = insertelement <16 x float> undef, float %69, i32 0
  %71 = shufflevector <16 x float> %70, <16 x float> undef, <16 x i32> zeroinitializer
  %72 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %71, <16 x float> %59, <16 x float> %compute.global1.sroa.68.070)
  %73 = add nuw nsw i64 %35, 1536
  %74 = getelementptr inbounds float, float* %p02, i64 %73
  %75 = load float, float* %74, align 64, !tbaa !198
  %76 = insertelement <16 x float> undef, float %75, i32 0
  %77 = shufflevector <16 x float> %76, <16 x float> undef, <16 x i32> zeroinitializer
  %78 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %77, <16 x float> %59, <16 x float> %compute.global1.sroa.102.069)
  %79 = or i64 %35, 1
  %80 = getelementptr inbounds float, float* %p02, i64 %79
  %81 = load float, float* %80, align 4, !tbaa !198
  %82 = insertelement <16 x float> undef, float %81, i32 0
  %83 = shufflevector <16 x float> %82, <16 x float> undef, <16 x i32> zeroinitializer
  %84 = getelementptr inbounds float, float* %p13, i64 %49
  %85 = bitcast float* %84 to <16 x float>*
  %86 = load <16 x float>, <16 x float>* %85, align 64, !tbaa !200
  %87 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %83, <16 x float> %86, <16 x float> %60)
  %88 = add nuw nsw i64 %35, 513
  %89 = getelementptr inbounds float, float* %p02, i64 %88
  %90 = load float, float* %89, align 4, !tbaa !198
  %91 = insertelement <16 x float> undef, float %90, i32 0
  %92 = shufflevector <16 x float> %91, <16 x float> undef, <16 x i32> zeroinitializer
  %93 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %92, <16 x float> %86, <16 x float> %66)
  %94 = add nuw nsw i64 %35, 1025
  %95 = getelementptr inbounds float, float* %p02, i64 %94
  %96 = load float, float* %95, align 4, !tbaa !198
  %97 = insertelement <16 x float> undef, float %96, i32 0
  %98 = shufflevector <16 x float> %97, <16 x float> undef, <16 x i32> zeroinitializer
  %99 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %98, <16 x float> %86, <16 x float> %72)
  %100 = add nuw nsw i64 %35, 1537
  %101 = getelementptr inbounds float, float* %p02, i64 %100
  %102 = load float, float* %101, align 4, !tbaa !198
  %103 = insertelement <16 x float> undef, float %102, i32 0
  %104 = shufflevector <16 x float> %103, <16 x float> undef, <16 x i32> zeroinitializer
  %105 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %104, <16 x float> %86, <16 x float> %78)
  %106 = or i64 %35, 2
  %107 = getelementptr inbounds float, float* %p02, i64 %106
  %108 = load float, float* %107, align 8, !tbaa !198
  %109 = insertelement <16 x float> undef, float %108, i32 0
  %110 = shufflevector <16 x float> %109, <16 x float> undef, <16 x i32> zeroinitializer
  %111 = getelementptr inbounds float, float* %p13, i64 %42
  %112 = bitcast float* %111 to <16 x float>*
  %113 = load <16 x float>, <16 x float>* %112, align 64, !tbaa !200
  %114 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %110, <16 x float> %113, <16 x float> %87)
  %115 = add nuw nsw i64 %35, 514
  %116 = getelementptr inbounds float, float* %p02, i64 %115
  %117 = load float, float* %116, align 8, !tbaa !198
  %118 = insertelement <16 x float> undef, float %117, i32 0
  %119 = shufflevector <16 x float> %118, <16 x float> undef, <16 x i32> zeroinitializer
  %120 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %119, <16 x float> %113, <16 x float> %93)
  %121 = add nuw nsw i64 %35, 1026
  %122 = getelementptr inbounds float, float* %p02, i64 %121
  %123 = load float, float* %122, align 8, !tbaa !198
  %124 = insertelement <16 x float> undef, float %123, i32 0
  %125 = shufflevector <16 x float> %124, <16 x float> undef, <16 x i32> zeroinitializer
  %126 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %125, <16 x float> %113, <16 x float> %99)
  %127 = add nuw nsw i64 %35, 1538
  %128 = getelementptr inbounds float, float* %p02, i64 %127
  %129 = load float, float* %128, align 8, !tbaa !198
  %130 = insertelement <16 x float> undef, float %129, i32 0
  %131 = shufflevector <16 x float> %130, <16 x float> undef, <16 x i32> zeroinitializer
  %132 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %131, <16 x float> %113, <16 x float> %105)
  %133 = or i64 %35, 3
  %134 = getelementptr inbounds float, float* %p02, i64 %133
  %135 = load float, float* %134, align 4, !tbaa !198
  %136 = insertelement <16 x float> undef, float %135, i32 0
  %137 = shufflevector <16 x float> %136, <16 x float> undef, <16 x i32> zeroinitializer
  %138 = getelementptr inbounds float, float* %p13, i64 %41
  %139 = bitcast float* %138 to <16 x float>*
  %140 = load <16 x float>, <16 x float>* %139, align 64, !tbaa !200
  %141 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %137, <16 x float> %140, <16 x float> %114)
  %142 = add nuw nsw i64 %35, 515
  %143 = getelementptr inbounds float, float* %p02, i64 %142
  %144 = load float, float* %143, align 4, !tbaa !198
  %145 = insertelement <16 x float> undef, float %144, i32 0
  %146 = shufflevector <16 x float> %145, <16 x float> undef, <16 x i32> zeroinitializer
  %147 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %146, <16 x float> %140, <16 x float> %120)
  %148 = add nuw nsw i64 %35, 1027
  %149 = getelementptr inbounds float, float* %p02, i64 %148
  %150 = load float, float* %149, align 4, !tbaa !198
  %151 = insertelement <16 x float> undef, float %150, i32 0
  %152 = shufflevector <16 x float> %151, <16 x float> undef, <16 x i32> zeroinitializer
  %153 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %152, <16 x float> %140, <16 x float> %126)
  %154 = add nuw nsw i64 %35, 1539
  %155 = getelementptr inbounds float, float* %p02, i64 %154
  %156 = load float, float* %155, align 4, !tbaa !198
  %157 = insertelement <16 x float> undef, float %156, i32 0
  %158 = shufflevector <16 x float> %157, <16 x float> undef, <16 x i32> zeroinitializer
  %159 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %158, <16 x float> %140, <16 x float> %132)
  %160 = or i64 %35, 4
  %161 = getelementptr inbounds float, float* %p02, i64 %160
  %162 = load float, float* %161, align 16, !tbaa !198
  %163 = insertelement <16 x float> undef, float %162, i32 0
  %164 = shufflevector <16 x float> %163, <16 x float> undef, <16 x i32> zeroinitializer
  %165 = getelementptr inbounds float, float* %p13, i64 %40
  %166 = bitcast float* %165 to <16 x float>*
  %167 = load <16 x float>, <16 x float>* %166, align 64, !tbaa !200
  %168 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %164, <16 x float> %167, <16 x float> %141)
  %169 = add nuw nsw i64 %35, 516
  %170 = getelementptr inbounds float, float* %p02, i64 %169
  %171 = load float, float* %170, align 16, !tbaa !198
  %172 = insertelement <16 x float> undef, float %171, i32 0
  %173 = shufflevector <16 x float> %172, <16 x float> undef, <16 x i32> zeroinitializer
  %174 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %173, <16 x float> %167, <16 x float> %147)
  %175 = add nuw nsw i64 %35, 1028
  %176 = getelementptr inbounds float, float* %p02, i64 %175
  %177 = load float, float* %176, align 16, !tbaa !198
  %178 = insertelement <16 x float> undef, float %177, i32 0
  %179 = shufflevector <16 x float> %178, <16 x float> undef, <16 x i32> zeroinitializer
  %180 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %179, <16 x float> %167, <16 x float> %153)
  %181 = add nuw nsw i64 %35, 1540
  %182 = getelementptr inbounds float, float* %p02, i64 %181
  %183 = load float, float* %182, align 16, !tbaa !198
  %184 = insertelement <16 x float> undef, float %183, i32 0
  %185 = shufflevector <16 x float> %184, <16 x float> undef, <16 x i32> zeroinitializer
  %186 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %185, <16 x float> %167, <16 x float> %159)
  %187 = or i64 %35, 5
  %188 = getelementptr inbounds float, float* %p02, i64 %187
  %189 = load float, float* %188, align 4, !tbaa !198
  %190 = insertelement <16 x float> undef, float %189, i32 0
  %191 = shufflevector <16 x float> %190, <16 x float> undef, <16 x i32> zeroinitializer
  %192 = getelementptr inbounds float, float* %p13, i64 %39
  %193 = bitcast float* %192 to <16 x float>*
  %194 = load <16 x float>, <16 x float>* %193, align 64, !tbaa !200
  %195 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %191, <16 x float> %194, <16 x float> %168)
  %196 = add nuw nsw i64 %35, 517
  %197 = getelementptr inbounds float, float* %p02, i64 %196
  %198 = load float, float* %197, align 4, !tbaa !198
  %199 = insertelement <16 x float> undef, float %198, i32 0
  %200 = shufflevector <16 x float> %199, <16 x float> undef, <16 x i32> zeroinitializer
  %201 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %200, <16 x float> %194, <16 x float> %174)
  %202 = add nuw nsw i64 %35, 1029
  %203 = getelementptr inbounds float, float* %p02, i64 %202
  %204 = load float, float* %203, align 4, !tbaa !198
  %205 = insertelement <16 x float> undef, float %204, i32 0
  %206 = shufflevector <16 x float> %205, <16 x float> undef, <16 x i32> zeroinitializer
  %207 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %206, <16 x float> %194, <16 x float> %180)
  %208 = add nuw nsw i64 %35, 1541
  %209 = getelementptr inbounds float, float* %p02, i64 %208
  %210 = load float, float* %209, align 4, !tbaa !198
  %211 = insertelement <16 x float> undef, float %210, i32 0
  %212 = shufflevector <16 x float> %211, <16 x float> undef, <16 x i32> zeroinitializer
  %213 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %212, <16 x float> %194, <16 x float> %186)
  %214 = or i64 %35, 6
  %215 = getelementptr inbounds float, float* %p02, i64 %214
  %216 = load float, float* %215, align 8, !tbaa !198
  %217 = insertelement <16 x float> undef, float %216, i32 0
  %218 = shufflevector <16 x float> %217, <16 x float> undef, <16 x i32> zeroinitializer
  %219 = getelementptr inbounds float, float* %p13, i64 %38
  %220 = bitcast float* %219 to <16 x float>*
  %221 = load <16 x float>, <16 x float>* %220, align 64, !tbaa !200
  %222 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %218, <16 x float> %221, <16 x float> %195)
  %223 = add nuw nsw i64 %35, 518
  %224 = getelementptr inbounds float, float* %p02, i64 %223
  %225 = load float, float* %224, align 8, !tbaa !198
  %226 = insertelement <16 x float> undef, float %225, i32 0
  %227 = shufflevector <16 x float> %226, <16 x float> undef, <16 x i32> zeroinitializer
  %228 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %227, <16 x float> %221, <16 x float> %201)
  %229 = add nuw nsw i64 %35, 1030
  %230 = getelementptr inbounds float, float* %p02, i64 %229
  %231 = load float, float* %230, align 8, !tbaa !198
  %232 = insertelement <16 x float> undef, float %231, i32 0
  %233 = shufflevector <16 x float> %232, <16 x float> undef, <16 x i32> zeroinitializer
  %234 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %233, <16 x float> %221, <16 x float> %207)
  %235 = add nuw nsw i64 %35, 1542
  %236 = getelementptr inbounds float, float* %p02, i64 %235
  %237 = load float, float* %236, align 8, !tbaa !198
  %238 = insertelement <16 x float> undef, float %237, i32 0
  %239 = shufflevector <16 x float> %238, <16 x float> undef, <16 x i32> zeroinitializer
  %240 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %239, <16 x float> %221, <16 x float> %213)
  %241 = or i64 %35, 7
  %242 = getelementptr inbounds float, float* %p02, i64 %241
  %243 = load float, float* %242, align 4, !tbaa !198
  %244 = insertelement <16 x float> undef, float %243, i32 0
  %245 = shufflevector <16 x float> %244, <16 x float> undef, <16 x i32> zeroinitializer
  %246 = getelementptr inbounds float, float* %p13, i64 %52
  %247 = bitcast float* %246 to <16 x float>*
  %248 = load <16 x float>, <16 x float>* %247, align 64, !tbaa !200
  %249 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %245, <16 x float> %248, <16 x float> %222)
  %250 = add nuw nsw i64 %35, 519
  %251 = getelementptr inbounds float, float* %p02, i64 %250
  %252 = load float, float* %251, align 4, !tbaa !198
  %253 = insertelement <16 x float> undef, float %252, i32 0
  %254 = shufflevector <16 x float> %253, <16 x float> undef, <16 x i32> zeroinitializer
  %255 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %254, <16 x float> %248, <16 x float> %228)
  %256 = add nuw nsw i64 %35, 1031
  %257 = getelementptr inbounds float, float* %p02, i64 %256
  %258 = load float, float* %257, align 4, !tbaa !198
  %259 = insertelement <16 x float> undef, float %258, i32 0
  %260 = shufflevector <16 x float> %259, <16 x float> undef, <16 x i32> zeroinitializer
  %261 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %260, <16 x float> %248, <16 x float> %234)
  %262 = add nuw nsw i64 %35, 1543
  %263 = getelementptr inbounds float, float* %p02, i64 %262
  %264 = load float, float* %263, align 4, !tbaa !198
  %265 = insertelement <16 x float> undef, float %264, i32 0
  %266 = shufflevector <16 x float> %265, <16 x float> undef, <16 x i32> zeroinitializer
  %267 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %266, <16 x float> %248, <16 x float> %240)
  %268 = or i64 %35, 8
  %269 = getelementptr inbounds float, float* %p02, i64 %268
  %270 = load float, float* %269, align 32, !tbaa !198
  %271 = insertelement <16 x float> undef, float %270, i32 0
  %272 = shufflevector <16 x float> %271, <16 x float> undef, <16 x i32> zeroinitializer
  %273 = getelementptr inbounds float, float* %p13, i64 %51
  %274 = bitcast float* %273 to <16 x float>*
  %275 = load <16 x float>, <16 x float>* %274, align 64, !tbaa !200
  %276 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %272, <16 x float> %275, <16 x float> %249)
  %277 = add nuw nsw i64 %35, 520
  %278 = getelementptr inbounds float, float* %p02, i64 %277
  %279 = load float, float* %278, align 32, !tbaa !198
  %280 = insertelement <16 x float> undef, float %279, i32 0
  %281 = shufflevector <16 x float> %280, <16 x float> undef, <16 x i32> zeroinitializer
  %282 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %281, <16 x float> %275, <16 x float> %255)
  %283 = add nuw nsw i64 %35, 1032
  %284 = getelementptr inbounds float, float* %p02, i64 %283
  %285 = load float, float* %284, align 32, !tbaa !198
  %286 = insertelement <16 x float> undef, float %285, i32 0
  %287 = shufflevector <16 x float> %286, <16 x float> undef, <16 x i32> zeroinitializer
  %288 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %287, <16 x float> %275, <16 x float> %261)
  %289 = add nuw nsw i64 %35, 1544
  %290 = getelementptr inbounds float, float* %p02, i64 %289
  %291 = load float, float* %290, align 32, !tbaa !198
  %292 = insertelement <16 x float> undef, float %291, i32 0
  %293 = shufflevector <16 x float> %292, <16 x float> undef, <16 x i32> zeroinitializer
  %294 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %293, <16 x float> %275, <16 x float> %267)
  %295 = or i64 %35, 9
  %296 = getelementptr inbounds float, float* %p02, i64 %295
  %297 = load float, float* %296, align 4, !tbaa !198
  %298 = insertelement <16 x float> undef, float %297, i32 0
  %299 = shufflevector <16 x float> %298, <16 x float> undef, <16 x i32> zeroinitializer
  %300 = getelementptr inbounds float, float* %p13, i64 %50
  %301 = bitcast float* %300 to <16 x float>*
  %302 = load <16 x float>, <16 x float>* %301, align 64, !tbaa !200
  %303 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %299, <16 x float> %302, <16 x float> %276)
  %304 = add nuw nsw i64 %35, 521
  %305 = getelementptr inbounds float, float* %p02, i64 %304
  %306 = load float, float* %305, align 4, !tbaa !198
  %307 = insertelement <16 x float> undef, float %306, i32 0
  %308 = shufflevector <16 x float> %307, <16 x float> undef, <16 x i32> zeroinitializer
  %309 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %308, <16 x float> %302, <16 x float> %282)
  %310 = add nuw nsw i64 %35, 1033
  %311 = getelementptr inbounds float, float* %p02, i64 %310
  %312 = load float, float* %311, align 4, !tbaa !198
  %313 = insertelement <16 x float> undef, float %312, i32 0
  %314 = shufflevector <16 x float> %313, <16 x float> undef, <16 x i32> zeroinitializer
  %315 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %314, <16 x float> %302, <16 x float> %288)
  %316 = add nuw nsw i64 %35, 1545
  %317 = getelementptr inbounds float, float* %p02, i64 %316
  %318 = load float, float* %317, align 4, !tbaa !198
  %319 = insertelement <16 x float> undef, float %318, i32 0
  %320 = shufflevector <16 x float> %319, <16 x float> undef, <16 x i32> zeroinitializer
  %321 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %320, <16 x float> %302, <16 x float> %294)
  %322 = or i64 %35, 10
  %323 = getelementptr inbounds float, float* %p02, i64 %322
  %324 = load float, float* %323, align 8, !tbaa !198
  %325 = insertelement <16 x float> undef, float %324, i32 0
  %326 = shufflevector <16 x float> %325, <16 x float> undef, <16 x i32> zeroinitializer
  %327 = getelementptr inbounds float, float* %p13, i64 %48
  %328 = bitcast float* %327 to <16 x float>*
  %329 = load <16 x float>, <16 x float>* %328, align 64, !tbaa !200
  %330 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %326, <16 x float> %329, <16 x float> %303)
  %331 = add nuw nsw i64 %35, 522
  %332 = getelementptr inbounds float, float* %p02, i64 %331
  %333 = load float, float* %332, align 8, !tbaa !198
  %334 = insertelement <16 x float> undef, float %333, i32 0
  %335 = shufflevector <16 x float> %334, <16 x float> undef, <16 x i32> zeroinitializer
  %336 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %335, <16 x float> %329, <16 x float> %309)
  %337 = add nuw nsw i64 %35, 1034
  %338 = getelementptr inbounds float, float* %p02, i64 %337
  %339 = load float, float* %338, align 8, !tbaa !198
  %340 = insertelement <16 x float> undef, float %339, i32 0
  %341 = shufflevector <16 x float> %340, <16 x float> undef, <16 x i32> zeroinitializer
  %342 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %341, <16 x float> %329, <16 x float> %315)
  %343 = add nuw nsw i64 %35, 1546
  %344 = getelementptr inbounds float, float* %p02, i64 %343
  %345 = load float, float* %344, align 8, !tbaa !198
  %346 = insertelement <16 x float> undef, float %345, i32 0
  %347 = shufflevector <16 x float> %346, <16 x float> undef, <16 x i32> zeroinitializer
  %348 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %347, <16 x float> %329, <16 x float> %321)
  %349 = or i64 %35, 11
  %350 = getelementptr inbounds float, float* %p02, i64 %349
  %351 = load float, float* %350, align 4, !tbaa !198
  %352 = insertelement <16 x float> undef, float %351, i32 0
  %353 = shufflevector <16 x float> %352, <16 x float> undef, <16 x i32> zeroinitializer
  %354 = getelementptr inbounds float, float* %p13, i64 %47
  %355 = bitcast float* %354 to <16 x float>*
  %356 = load <16 x float>, <16 x float>* %355, align 64, !tbaa !200
  %357 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %353, <16 x float> %356, <16 x float> %330)
  %358 = add nuw nsw i64 %35, 523
  %359 = getelementptr inbounds float, float* %p02, i64 %358
  %360 = load float, float* %359, align 4, !tbaa !198
  %361 = insertelement <16 x float> undef, float %360, i32 0
  %362 = shufflevector <16 x float> %361, <16 x float> undef, <16 x i32> zeroinitializer
  %363 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %362, <16 x float> %356, <16 x float> %336)
  %364 = add nuw nsw i64 %35, 1035
  %365 = getelementptr inbounds float, float* %p02, i64 %364
  %366 = load float, float* %365, align 4, !tbaa !198
  %367 = insertelement <16 x float> undef, float %366, i32 0
  %368 = shufflevector <16 x float> %367, <16 x float> undef, <16 x i32> zeroinitializer
  %369 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %368, <16 x float> %356, <16 x float> %342)
  %370 = add nuw nsw i64 %35, 1547
  %371 = getelementptr inbounds float, float* %p02, i64 %370
  %372 = load float, float* %371, align 4, !tbaa !198
  %373 = insertelement <16 x float> undef, float %372, i32 0
  %374 = shufflevector <16 x float> %373, <16 x float> undef, <16 x i32> zeroinitializer
  %375 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %374, <16 x float> %356, <16 x float> %348)
  %376 = or i64 %35, 12
  %377 = getelementptr inbounds float, float* %p02, i64 %376
  %378 = load float, float* %377, align 16, !tbaa !198
  %379 = insertelement <16 x float> undef, float %378, i32 0
  %380 = shufflevector <16 x float> %379, <16 x float> undef, <16 x i32> zeroinitializer
  %381 = getelementptr inbounds float, float* %p13, i64 %46
  %382 = bitcast float* %381 to <16 x float>*
  %383 = load <16 x float>, <16 x float>* %382, align 64, !tbaa !200
  %384 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %380, <16 x float> %383, <16 x float> %357)
  %385 = add nuw nsw i64 %35, 524
  %386 = getelementptr inbounds float, float* %p02, i64 %385
  %387 = load float, float* %386, align 16, !tbaa !198
  %388 = insertelement <16 x float> undef, float %387, i32 0
  %389 = shufflevector <16 x float> %388, <16 x float> undef, <16 x i32> zeroinitializer
  %390 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %389, <16 x float> %383, <16 x float> %363)
  %391 = add nuw nsw i64 %35, 1036
  %392 = getelementptr inbounds float, float* %p02, i64 %391
  %393 = load float, float* %392, align 16, !tbaa !198
  %394 = insertelement <16 x float> undef, float %393, i32 0
  %395 = shufflevector <16 x float> %394, <16 x float> undef, <16 x i32> zeroinitializer
  %396 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %395, <16 x float> %383, <16 x float> %369)
  %397 = add nuw nsw i64 %35, 1548
  %398 = getelementptr inbounds float, float* %p02, i64 %397
  %399 = load float, float* %398, align 16, !tbaa !198
  %400 = insertelement <16 x float> undef, float %399, i32 0
  %401 = shufflevector <16 x float> %400, <16 x float> undef, <16 x i32> zeroinitializer
  %402 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %401, <16 x float> %383, <16 x float> %375)
  %403 = or i64 %35, 13
  %404 = getelementptr inbounds float, float* %p02, i64 %403
  %405 = load float, float* %404, align 4, !tbaa !198
  %406 = insertelement <16 x float> undef, float %405, i32 0
  %407 = shufflevector <16 x float> %406, <16 x float> undef, <16 x i32> zeroinitializer
  %408 = getelementptr inbounds float, float* %p13, i64 %45
  %409 = bitcast float* %408 to <16 x float>*
  %410 = load <16 x float>, <16 x float>* %409, align 64, !tbaa !200
  %411 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %407, <16 x float> %410, <16 x float> %384)
  %412 = add nuw nsw i64 %35, 525
  %413 = getelementptr inbounds float, float* %p02, i64 %412
  %414 = load float, float* %413, align 4, !tbaa !198
  %415 = insertelement <16 x float> undef, float %414, i32 0
  %416 = shufflevector <16 x float> %415, <16 x float> undef, <16 x i32> zeroinitializer
  %417 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %416, <16 x float> %410, <16 x float> %390)
  %418 = add nuw nsw i64 %35, 1037
  %419 = getelementptr inbounds float, float* %p02, i64 %418
  %420 = load float, float* %419, align 4, !tbaa !198
  %421 = insertelement <16 x float> undef, float %420, i32 0
  %422 = shufflevector <16 x float> %421, <16 x float> undef, <16 x i32> zeroinitializer
  %423 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %422, <16 x float> %410, <16 x float> %396)
  %424 = add nuw nsw i64 %35, 1549
  %425 = getelementptr inbounds float, float* %p02, i64 %424
  %426 = load float, float* %425, align 4, !tbaa !198
  %427 = insertelement <16 x float> undef, float %426, i32 0
  %428 = shufflevector <16 x float> %427, <16 x float> undef, <16 x i32> zeroinitializer
  %429 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %428, <16 x float> %410, <16 x float> %402)
  %430 = or i64 %35, 14
  %431 = getelementptr inbounds float, float* %p02, i64 %430
  %432 = load float, float* %431, align 8, !tbaa !198
  %433 = insertelement <16 x float> undef, float %432, i32 0
  %434 = shufflevector <16 x float> %433, <16 x float> undef, <16 x i32> zeroinitializer
  %435 = getelementptr inbounds float, float* %p13, i64 %44
  %436 = bitcast float* %435 to <16 x float>*
  %437 = load <16 x float>, <16 x float>* %436, align 64, !tbaa !200
  %438 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %434, <16 x float> %437, <16 x float> %411)
  %439 = add nuw nsw i64 %35, 526
  %440 = getelementptr inbounds float, float* %p02, i64 %439
  %441 = load float, float* %440, align 8, !tbaa !198
  %442 = insertelement <16 x float> undef, float %441, i32 0
  %443 = shufflevector <16 x float> %442, <16 x float> undef, <16 x i32> zeroinitializer
  %444 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %443, <16 x float> %437, <16 x float> %417)
  %445 = add nuw nsw i64 %35, 1038
  %446 = getelementptr inbounds float, float* %p02, i64 %445
  %447 = load float, float* %446, align 8, !tbaa !198
  %448 = insertelement <16 x float> undef, float %447, i32 0
  %449 = shufflevector <16 x float> %448, <16 x float> undef, <16 x i32> zeroinitializer
  %450 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %449, <16 x float> %437, <16 x float> %423)
  %451 = add nuw nsw i64 %35, 1550
  %452 = getelementptr inbounds float, float* %p02, i64 %451
  %453 = load float, float* %452, align 8, !tbaa !198
  %454 = insertelement <16 x float> undef, float %453, i32 0
  %455 = shufflevector <16 x float> %454, <16 x float> undef, <16 x i32> zeroinitializer
  %456 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %455, <16 x float> %437, <16 x float> %429)
  %457 = or i64 %35, 15
  %458 = getelementptr inbounds float, float* %p02, i64 %457
  %459 = load float, float* %458, align 4, !tbaa !198
  %460 = insertelement <16 x float> undef, float %459, i32 0
  %461 = shufflevector <16 x float> %460, <16 x float> undef, <16 x i32> zeroinitializer
  %462 = getelementptr inbounds float, float* %p13, i64 %43
  %463 = bitcast float* %462 to <16 x float>*
  %464 = load <16 x float>, <16 x float>* %463, align 64, !tbaa !200
  %465 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %461, <16 x float> %464, <16 x float> %438)
  %466 = add nuw nsw i64 %35, 527
  %467 = getelementptr inbounds float, float* %p02, i64 %466
  %468 = load float, float* %467, align 4, !tbaa !198
  %469 = insertelement <16 x float> undef, float %468, i32 0
  %470 = shufflevector <16 x float> %469, <16 x float> undef, <16 x i32> zeroinitializer
  %471 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %470, <16 x float> %464, <16 x float> %444)
  %472 = add nuw nsw i64 %35, 1039
  %473 = getelementptr inbounds float, float* %p02, i64 %472
  %474 = load float, float* %473, align 4, !tbaa !198
  %475 = insertelement <16 x float> undef, float %474, i32 0
  %476 = shufflevector <16 x float> %475, <16 x float> undef, <16 x i32> zeroinitializer
  %477 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %476, <16 x float> %464, <16 x float> %450)
  %478 = add nuw nsw i64 %35, 1551
  %479 = getelementptr inbounds float, float* %p02, i64 %478
  %480 = load float, float* %479, align 4, !tbaa !198
  %481 = insertelement <16 x float> undef, float %480, i32 0
  %482 = shufflevector <16 x float> %481, <16 x float> undef, <16 x i32> zeroinitializer
  %483 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %482, <16 x float> %464, <16 x float> %456)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 32
  br i1 %exitcond, label %for_end_k.outer, label %for_body_k.outer, !prof !48

for_end_k.outer:                                  ; preds = %for_body_k.outer
  %484 = shl nuw nsw i32 %23, 4
  %cse_var_1 = or i32 %27, %484
  %485 = sext i32 %cse_var_1 to i64
  %486 = getelementptr inbounds float, float* %compute4, i64 %485
  %487 = bitcast float* %486 to <16 x float>*
  store <16 x float> %465, <16 x float>* %487, align 64, !tbaa !202
  %488 = add nsw i32 %cse_var_1, 512
  %489 = sext i32 %488 to i64
  %490 = getelementptr inbounds float, float* %compute4, i64 %489
  %491 = bitcast float* %490 to <16 x float>*
  store <16 x float> %471, <16 x float>* %491, align 64, !tbaa !202
  %492 = add nsw i32 %cse_var_1, 1024
  %493 = sext i32 %492 to i64
  %494 = getelementptr inbounds float, float* %compute4, i64 %493
  %495 = bitcast float* %494 to <16 x float>*
  store <16 x float> %477, <16 x float>* %495, align 64, !tbaa !202
  %496 = add nsw i32 %cse_var_1, 1536
  %497 = sext i32 %496 to i64
  %498 = getelementptr inbounds float, float* %compute4, i64 %497
  %499 = bitcast float* %498 to <16 x float>*
  store <16 x float> %483, <16 x float>* %499, align 64, !tbaa !202
  %500 = add nuw nsw i32 %y.inner.outer.x.inner.outer.fused74, 1
  %exitcond96 = icmp eq i32 %500, 128
  br i1 %exitcond96, label %for_end_y.inner.outer.x.inner.outer.fused, label %for_body_y.inner.outer.x.inner.outer.fused, !prof !48
}

; Function Attrs: nounwind readnone speculatable willreturn
declare <16 x float> @llvm.fmuladd.v16f32(<16 x float>, <16 x float>, <16 x float>) #5

; Function Attrs: nounwind readnone
define weak dso_local i16 @__truncsfhf2(float %a0) local_unnamed_addr #6 section ".text.tvm.fp16.conv" {
b0:
  %v0 = bitcast float %a0 to i32
  %v1 = and i32 %v0, 2147483647
  %v2 = add nsw i32 %v1, -947912704
  %v3 = add nsw i32 %v1, -1199570944
  %v4 = icmp ult i32 %v2, %v3
  br i1 %v4, label %b1, label %b5

b1:                                               ; preds = %b0
  %v5 = lshr i32 %v0, 13
  %v6 = and i32 %v5, 65535
  %v7 = add nuw nsw i32 %v6, -114688
  %v8 = and i32 %v0, 8191
  %v9 = icmp ugt i32 %v8, 4096
  br i1 %v9, label %b2, label %b3

b2:                                               ; preds = %b1
  %v10 = add nuw nsw i32 %v6, -114687
  br label %b13

b3:                                               ; preds = %b1
  %v11 = icmp eq i32 %v8, 4096
  br i1 %v11, label %b4, label %b13

b4:                                               ; preds = %b3
  %v12 = and i32 %v7, 65535
  %v13 = and i32 %v5, 1
  %v14 = add nuw nsw i32 %v12, %v13
  br label %b13

b5:                                               ; preds = %b0
  %v15 = icmp ugt i32 %v1, 2139095040
  br i1 %v15, label %b6, label %b7

b6:                                               ; preds = %b5
  %v16 = lshr i32 %v0, 13
  %v17 = and i32 %v16, 511
  %v18 = or i32 %v17, 32256
  br label %b13

b7:                                               ; preds = %b5
  %v19 = icmp ugt i32 %v1, 1199570943
  br i1 %v19, label %b13, label %b8

b8:                                               ; preds = %b7
  %v20 = icmp ult i32 %v1, 754974720
  br i1 %v20, label %b13, label %b9

b9:                                               ; preds = %b8
  %v21 = lshr i32 %v1, 23
  %v22 = sub nsw i32 113, %v21
  %v23 = and i32 %v0, 8388607
  %v24 = or i32 %v23, 8388608
  %v25 = add nsw i32 %v21, -81
  %v26 = shl i32 %v24, %v25
  %v27 = icmp ne i32 %v26, 0
  %v28 = lshr i32 %v24, %v22
  %v29 = zext i1 %v27 to i32
  %v30 = lshr i32 %v28, 13
  %v31 = and i32 %v28, 8191
  %v32 = or i32 %v31, %v29
  %v33 = icmp ugt i32 %v32, 4096
  br i1 %v33, label %b10, label %b11

b10:                                              ; preds = %b9
  %v34 = add nuw nsw i32 %v30, 1
  br label %b13

b11:                                              ; preds = %b9
  %v35 = icmp eq i32 %v32, 4096
  br i1 %v35, label %b12, label %b13

b12:                                              ; preds = %b11
  %v36 = and i32 %v30, 1
  %v37 = add nuw nsw i32 %v36, %v30
  br label %b13

b13:                                              ; preds = %b12, %b11, %b10, %b8, %b7, %b6, %b4, %b3, %b2
  %v38 = phi i32 [ %v18, %b6 ], [ %v10, %b2 ], [ %v14, %b4 ], [ %v7, %b3 ], [ 31744, %b7 ], [ 0, %b8 ], [ %v34, %b10 ], [ %v37, %b12 ], [ %v30, %b11 ]
  %v39 = lshr i32 %v0, 16
  %v40 = and i32 %v39, 32768
  %v41 = or i32 %v38, %v40
  %vlast = trunc i32 %v41 to i16
  ret i16 %vlast
}

; Function Attrs: nounwind readnone
define weak dso_local float @__extendhfsf2(i16 %a0) local_unnamed_addr #6 section ".text.tvm.fp16.conv" {
b0:
  %v1 = and i16 %a0, 32767
  %v2 = zext i16 %v1 to i32
  %v3 = add nsw i16 %v1, -1024
  %v4 = icmp ult i16 %v3, 30720
  br i1 %v4, label %b1, label %b2

b1:                                               ; preds = %b0
  %v5 = shl nuw nsw i32 %v2, 13
  %v6 = add nuw nsw i32 %v5, 939524096
  br label %b6

b2:                                               ; preds = %b0
  %v7 = icmp ugt i16 %v1, 31743
  br i1 %v7, label %b3, label %b4

b3:                                               ; preds = %b2
  %v8 = shl nuw nsw i32 %v2, 13
  %v9 = or i32 %v8, 2139095040
  br label %b6

b4:                                               ; preds = %b2
  %v10 = icmp eq i16 %v1, 0
  br i1 %v10, label %b6, label %b5

b5:                                               ; preds = %b4
  %v11 = icmp ult i16 %v1, 256
  %v12 = lshr i32 %v2, 8
  %v13 = select i1 %v11, i32 %v2, i32 %v12
  %v14 = select i1 %v11, i32 32, i32 24
  %v15 = icmp ult i32 %v13, 16
  %v16 = lshr i32 %v13, 4
  %v17 = add nsw i32 %v14, -4
  %v18 = select i1 %v15, i32 %v13, i32 %v16
  %v19 = select i1 %v15, i32 %v14, i32 %v17
  %v20 = icmp ult i32 %v18, 4
  %v21 = lshr i32 %v18, 2
  %v22 = add nsw i32 %v19, -2
  %v23 = select i1 %v20, i32 %v18, i32 %v21
  %v24 = select i1 %v20, i32 %v19, i32 %v22
  %v25 = icmp ult i32 %v23, 2
  %v26 = sub nsw i32 0, %v23
  %v27 = select i1 %v25, i32 %v26, i32 -2
  %v28 = add nsw i32 %v27, %v24
  %v29 = add nsw i32 %v28, -8
  %v30 = shl i32 %v2, %v29
  %v31 = xor i32 %v30, 8388608
  %v32 = shl i32 %v28, 23
  %v33 = sub i32 1124073472, %v32
  %v34 = or i32 %v31, %v33
  br label %b6

b6:                                               ; preds = %b5, %b4, %b3, %b1
  %v35 = phi i32 [ %v6, %b1 ], [ %v9, %b3 ], [ %v34, %b5 ], [ 0, %b4 ]
  %v36 = and i16 %a0, -32768
  %v37 = zext i16 %v36 to i32
  %v38 = shl nuw i32 %v37, 16
  %v39 = or i32 %v35, %v38
  %v40 = bitcast i32 %v39 to float
  ret float %v40
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #5

attributes #0 = { "target-cpu"="core-avx2" }
attributes #1 = { nounwind willreturn }
attributes #2 = { noinline "target-cpu"="core-avx2" }
attributes #3 = { nofree norecurse nounwind "target-cpu"="core-avx2" }
attributes #4 = { nofree nounwind "target-cpu"="core-avx2" }
attributes #5 = { nounwind readnone speculatable willreturn }
attributes #6 = { nounwind readnone "target-cpu"="core-avx2" "target-features" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "TVM", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "main.tir", directory: ".")
!2 = !{}
!3 = !{i32 2, !"tvm_target", !"llvm -mtriple=x86_64-pc-linux-gnu -mcpu=core-avx2"}
!4 = !{i32 4, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "main.tir", scope: !1, file: !1, type: !6, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9, !10, !8, !9, !10, !9}
!8 = !DIBasicType(name: "int32", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8)
!11 = !{!12, !13, !14, !15, !16, !17}
!12 = !DILocalVariable(name: "arg1", arg: 1, scope: !5, file: !1, type: !9)
!13 = !DILocalVariable(name: "arg2", arg: 2, scope: !5, file: !1, type: !10)
!14 = !DILocalVariable(name: "arg3", arg: 3, scope: !5, file: !1, type: !8)
!15 = !DILocalVariable(name: "arg4", arg: 4, scope: !5, file: !1, type: !9)
!16 = !DILocalVariable(name: "arg5", arg: 5, scope: !5, file: !1, type: !10)
!17 = !DILocalVariable(name: "arg6", arg: 6, scope: !5, file: !1, type: !9)
!18 = !DILocation(line: 0, scope: !5)
!19 = !{!"branch_weights", i32 1048576, i32 1}
!20 = !{!21, !21, i64 0}
!21 = !{!"ctx_ptr", !22, i64 0}
!22 = !{!"tvm-tbaa"}
!23 = !{!24, !24, i64 0}
!24 = !{!"0x2c17290.w4.b0", !25, i64 0}
!25 = !{!"0x2c17290.w8.b0", !26, i64 0}
!26 = !{!"0x2c17290.w16.b0", !27, i64 0}
!27 = !{!"0x2c17290.w32.b0", !28, i64 0}
!28 = !{!"0x2c17290.w64.b0", !29, i64 0}
!29 = !{!"0x2c17290.w128.b0", !30, i64 0}
!30 = !{!"0x2c17290.w256.b0", !31, i64 0}
!31 = !{!"0x2c17290.w512.b0", !32, i64 0}
!32 = !{!"0x2c17290.w1024.b0", !33, i64 0}
!33 = !{!"0x2c17290", !22, i64 0}
!34 = !{!35, !35, i64 0}
!35 = !{!"0x2c17290.w4.b4", !25, i64 0}
!36 = !{!37, !37, i64 0}
!37 = !{!"0x2ee67a0.w8.b0", !38, i64 0}
!38 = !{!"0x2ee67a0.w16.b0", !39, i64 0}
!39 = !{!"0x2ee67a0.w32.b0", !40, i64 0}
!40 = !{!"0x2ee67a0.w64.b0", !41, i64 0}
!41 = !{!"0x2ee67a0.w128.b0", !42, i64 0}
!42 = !{!"0x2ee67a0.w256.b0", !43, i64 0}
!43 = !{!"0x2ee67a0.w512.b0", !44, i64 0}
!44 = !{!"0x2ee67a0.w1024.b0", !45, i64 0}
!45 = !{!"0x2ee67a0", !22, i64 0}
!46 = !{!47, !47, i64 0}
!47 = !{!"0x2ee67a0.w8.b8", !38, i64 0}
!48 = !{!"branch_weights", i32 1, i32 1048576}
!49 = !{!50, !50, i64 0}
!50 = !{!"0x2b0bde0.w8.b0", !51, i64 0}
!51 = !{!"0x2b0bde0.w16.b0", !52, i64 0}
!52 = !{!"0x2b0bde0.w32.b0", !53, i64 0}
!53 = !{!"0x2b0bde0.w64.b0", !54, i64 0}
!54 = !{!"0x2b0bde0.w128.b0", !55, i64 0}
!55 = !{!"0x2b0bde0.w256.b0", !56, i64 0}
!56 = !{!"0x2b0bde0.w512.b0", !57, i64 0}
!57 = !{!"0x2b0bde0.w1024.b0", !58, i64 0}
!58 = !{!"0x2b0bde0", !22, i64 0}
!59 = !{!60, !60, i64 0}
!60 = !{!"0x2b0bde0.w8.b8", !51, i64 0}
!61 = !{!62, !62, i64 0}
!62 = !{!"0x31b1b60.w8.b0", !63, i64 0}
!63 = !{!"0x31b1b60.w16.b0", !64, i64 0}
!64 = !{!"0x31b1b60.w32.b0", !65, i64 0}
!65 = !{!"0x31b1b60.w64.b0", !66, i64 0}
!66 = !{!"0x31b1b60.w128.b0", !67, i64 0}
!67 = !{!"0x31b1b60.w256.b0", !68, i64 0}
!68 = !{!"0x31b1b60.w512.b0", !69, i64 0}
!69 = !{!"0x31b1b60.w1024.b0", !70, i64 0}
!70 = !{!"0x31b1b60", !22, i64 0}
!71 = !{!72, !72, i64 0}
!72 = !{!"0x31b1b60.w8.b8", !63, i64 0}
!73 = !{!74, !74, i64 0}
!74 = !{!"0x31b1b60.w8.b16", !75, i64 0}
!75 = !{!"0x31b1b60.w16.b16", !64, i64 0}
!76 = !{!77, !77, i64 0}
!77 = !{!"0x31b0810.w8.b0", !78, i64 0}
!78 = !{!"0x31b0810.w16.b0", !79, i64 0}
!79 = !{!"0x31b0810.w32.b0", !80, i64 0}
!80 = !{!"0x31b0810.w64.b0", !81, i64 0}
!81 = !{!"0x31b0810.w128.b0", !82, i64 0}
!82 = !{!"0x31b0810.w256.b0", !83, i64 0}
!83 = !{!"0x31b0810.w512.b0", !84, i64 0}
!84 = !{!"0x31b0810.w1024.b0", !85, i64 0}
!85 = !{!"0x31b0810", !22, i64 0}
!86 = !{!87, !87, i64 0}
!87 = !{!"0x31b0810.w8.b8", !78, i64 0}
!88 = !{!89, !89, i64 0}
!89 = !{!"0x31b0810.w8.b16", !90, i64 0}
!90 = !{!"0x31b0810.w16.b16", !79, i64 0}
!91 = !{!92, !92, i64 0}
!92 = !{!"0x3184b40", !22, i64 0}
!93 = !{!94, !94, i64 0}
!94 = !{!"0x256e700", !22, i64 0}
!95 = distinct !DISubprogram(name: "main.tir", scope: !1, file: !1, type: !6, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !96)
!96 = !{!97, !98, !99, !100, !101, !102}
!97 = !DILocalVariable(name: "arg1", arg: 1, scope: !95, file: !1, type: !9)
!98 = !DILocalVariable(name: "arg2", arg: 2, scope: !95, file: !1, type: !10)
!99 = !DILocalVariable(name: "arg3", arg: 3, scope: !95, file: !1, type: !8)
!100 = !DILocalVariable(name: "arg4", arg: 4, scope: !95, file: !1, type: !9)
!101 = !DILocalVariable(name: "arg5", arg: 5, scope: !95, file: !1, type: !10)
!102 = !DILocalVariable(name: "arg6", arg: 6, scope: !95, file: !1, type: !9)
!103 = !DILocation(line: 0, scope: !95)
!104 = !{!105, !105, i64 0}
!105 = !{!"0x304f6f0.w4.b0", !106, i64 0}
!106 = !{!"0x304f6f0.w8.b0", !107, i64 0}
!107 = !{!"0x304f6f0.w16.b0", !108, i64 0}
!108 = !{!"0x304f6f0.w32.b0", !109, i64 0}
!109 = !{!"0x304f6f0.w64.b0", !110, i64 0}
!110 = !{!"0x304f6f0.w128.b0", !111, i64 0}
!111 = !{!"0x304f6f0.w256.b0", !112, i64 0}
!112 = !{!"0x304f6f0.w512.b0", !113, i64 0}
!113 = !{!"0x304f6f0.w1024.b0", !114, i64 0}
!114 = !{!"0x304f6f0", !22, i64 0}
!115 = !{!116, !116, i64 0}
!116 = !{!"0x304f6f0.w4.b4", !106, i64 0}
!117 = !{!118, !118, i64 0}
!118 = !{!"0x304f6f0.w4.b8", !119, i64 0}
!119 = !{!"0x304f6f0.w8.b8", !107, i64 0}
!120 = !{!121, !121, i64 0}
!121 = !{!"0x31b5f10.w8.b0", !122, i64 0}
!122 = !{!"0x31b5f10.w16.b0", !123, i64 0}
!123 = !{!"0x31b5f10.w32.b0", !124, i64 0}
!124 = !{!"0x31b5f10.w64.b0", !125, i64 0}
!125 = !{!"0x31b5f10.w128.b0", !126, i64 0}
!126 = !{!"0x31b5f10.w256.b0", !127, i64 0}
!127 = !{!"0x31b5f10.w512.b0", !128, i64 0}
!128 = !{!"0x31b5f10.w1024.b0", !129, i64 0}
!129 = !{!"0x31b5f10", !22, i64 0}
!130 = !{!131, !131, i64 0}
!131 = !{!"0x31b5f10.w8.b8", !122, i64 0}
!132 = !{!133, !133, i64 0}
!133 = !{!"0x2cf46e0.w8.b0", !134, i64 0}
!134 = !{!"0x2cf46e0.w16.b0", !135, i64 0}
!135 = !{!"0x2cf46e0.w32.b0", !136, i64 0}
!136 = !{!"0x2cf46e0.w64.b0", !137, i64 0}
!137 = !{!"0x2cf46e0.w128.b0", !138, i64 0}
!138 = !{!"0x2cf46e0.w256.b0", !139, i64 0}
!139 = !{!"0x2cf46e0.w512.b0", !140, i64 0}
!140 = !{!"0x2cf46e0.w1024.b0", !141, i64 0}
!141 = !{!"0x2cf46e0", !22, i64 0}
!142 = !{!143, !143, i64 0}
!143 = !{!"0x2cf46e0.w8.b8", !134, i64 0}
!144 = !{!145, !145, i64 0}
!145 = !{!"0x2cc4190.w8.b0", !146, i64 0}
!146 = !{!"0x2cc4190.w16.b0", !147, i64 0}
!147 = !{!"0x2cc4190.w32.b0", !148, i64 0}
!148 = !{!"0x2cc4190.w64.b0", !149, i64 0}
!149 = !{!"0x2cc4190.w128.b0", !150, i64 0}
!150 = !{!"0x2cc4190.w256.b0", !151, i64 0}
!151 = !{!"0x2cc4190.w512.b0", !152, i64 0}
!152 = !{!"0x2cc4190.w1024.b0", !153, i64 0}
!153 = !{!"0x2cc4190", !22, i64 0}
!154 = !{!155, !155, i64 0}
!155 = !{!"0x2cc4190.w8.b8", !146, i64 0}
!156 = !{!157, !157, i64 0}
!157 = !{!"0x2cc4190.w8.b16", !158, i64 0}
!158 = !{!"0x2cc4190.w16.b16", !147, i64 0}
!159 = !{!160, !160, i64 0}
!160 = !{!"0x2ee49a0.w8.b0", !161, i64 0}
!161 = !{!"0x2ee49a0.w16.b0", !162, i64 0}
!162 = !{!"0x2ee49a0.w32.b0", !163, i64 0}
!163 = !{!"0x2ee49a0.w64.b0", !164, i64 0}
!164 = !{!"0x2ee49a0.w128.b0", !165, i64 0}
!165 = !{!"0x2ee49a0.w256.b0", !166, i64 0}
!166 = !{!"0x2ee49a0.w512.b0", !167, i64 0}
!167 = !{!"0x2ee49a0.w1024.b0", !168, i64 0}
!168 = !{!"0x2ee49a0", !22, i64 0}
!169 = !{!170, !170, i64 0}
!170 = !{!"0x2ee49a0.w8.b8", !161, i64 0}
!171 = !{!172, !172, i64 0}
!172 = !{!"0x2ee49a0.w8.b16", !173, i64 0}
!173 = !{!"0x2ee49a0.w16.b16", !162, i64 0}
!174 = !{!175, !175, i64 0}
!175 = !{!"0x31b6be0.w8.b0", !176, i64 0}
!176 = !{!"0x31b6be0.w16.b0", !177, i64 0}
!177 = !{!"0x31b6be0.w32.b0", !178, i64 0}
!178 = !{!"0x31b6be0.w64.b0", !179, i64 0}
!179 = !{!"0x31b6be0.w128.b0", !180, i64 0}
!180 = !{!"0x31b6be0.w256.b0", !181, i64 0}
!181 = !{!"0x31b6be0.w512.b0", !182, i64 0}
!182 = !{!"0x31b6be0.w1024.b0", !183, i64 0}
!183 = !{!"0x31b6be0", !22, i64 0}
!184 = !{!185, !185, i64 0}
!185 = !{!"0x31b6be0.w8.b8", !176, i64 0}
!186 = !{!187, !187, i64 0}
!187 = !{!"0x31b57b0.w8.b0", !188, i64 0}
!188 = !{!"0x31b57b0.w16.b0", !189, i64 0}
!189 = !{!"0x31b57b0.w32.b0", !190, i64 0}
!190 = !{!"0x31b57b0.w64.b0", !191, i64 0}
!191 = !{!"0x31b57b0.w128.b0", !192, i64 0}
!192 = !{!"0x31b57b0.w256.b0", !193, i64 0}
!193 = !{!"0x31b57b0.w512.b0", !194, i64 0}
!194 = !{!"0x31b57b0.w1024.b0", !195, i64 0}
!195 = !{!"0x31b57b0", !22, i64 0}
!196 = !{!197, !197, i64 0}
!197 = !{!"0x31b57b0.w8.b8", !188, i64 0}
!198 = !{!199, !199, i64 0}
!199 = !{!"0x2cd0c10", !22, i64 0}
!200 = !{!201, !201, i64 0}
!201 = !{!"0x2cd0bd0", !22, i64 0}
!202 = !{!203, !203, i64 0}
!203 = !{!"0x2e97d20", !22, i64 0}
