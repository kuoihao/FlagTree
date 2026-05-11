// RUN: triton-opt --triton-tle-lower-pipe-to-nvws --nvgpu-test-ws-lower-token %s | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pipe_to_mbarrier
  // CHECK-NOT: nvws.
  // CHECK: ttg.local_alloc
  // CHECK-SAME: !ttg.memdesc<2x1xi32
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK-COUNT-4: ttng.init_barrier {{.*}}, 128
  // CHECK: gpu.barrier
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 0>}
  // CHECK: ttng.arrive_barrier {{.*}}, 128 {async_task_id = array<i32: 0>, release_fence = true}
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 1>}
  // CHECK: ttg.local_load {{.*}} {async_task_id = array<i32: 1>}
  // CHECK: arith.cmpi ne
  // CHECK: ttng.arrive_barrier {{.*}}, 128 {async_task_id = array<i32: 1>}
  tt.func @pipe_to_mbarrier(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_acquire %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %a[%c0] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    %closed = tle.pipe.reader_wait %a[%c1, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    scf.if %closed {
    }
    tle.pipe.reader_release %a[%c1] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @pipe_cpasync_to_mbarrier
  // CHECK-NOT: nvws.
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 0>}
  // CHECK: ttng.async_copy_mbarrier_arrive {{.*}} {async_task_id = array<i32: 0>, noIncrement}
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 1>}
  tt.func @pipe_cpasync_to_mbarrier(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a_async", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_acquire %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a_async", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %a[%c0] {capacity = 2 : i32, pipe_name = "a_async", field_names = ["a"], scope = "cta", tle.pipe_commit_cp_async} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    %closed = tle.pipe.reader_wait %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a_async", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    scf.if %closed {
    }
    tt.return
  }

  // CHECK-LABEL: @pipe_tma_to_mbarrier
  // CHECK-NOT: nvws.
  // CHECK: ttng.init_barrier {{.*}}, 1
  // CHECK: ttng.init_barrier {{.*}}, 128
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 0>}
  // CHECK: ttng.barrier_expect {{.*}}, 8192
  // CHECK: ttng.async_tma_copy_global_to_local
  // CHECK-NOT: ttng.arrive_barrier {{.*}} {async_task_id = array<i32: 0>}
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 1>}
  tt.func @pipe_tma_to_mbarrier(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %a: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_commit %a[%c0] {capacity = 2 : i32, pipe_name = "a_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %closed = tle.pipe.reader_wait %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    scf.if %closed {
    }
    tt.return
  }

  // CHECK-LABEL: @one_shot_tma_pipe_to_mbarrier
  // CHECK-NOT: nvws.
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
  // CHECK: ttng.init_barrier {{.*}}, 1
  // CHECK-NOT: ttng.init_barrier {{.*}}, 128
  // CHECK: ttng.barrier_expect {{.*}}, 8192
  // CHECK: ttng.async_tma_copy_global_to_local
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 1>}
  // CHECK: arith.constant {{.*}}false
  // CHECK-NOT: ttng.arrive_barrier
  tt.func @one_shot_tma_pipe_to_mbarrier(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %a: !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 1 : i32, pipe_name = "one_shot_tma", field_names = ["a"], scope = "cta", one_shot = true} : !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %a[%c0, %false] {capacity = 1 : i32, pipe_name = "one_shot_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable>
    %slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_commit %a[%c0] {capacity = 1 : i32, pipe_name = "one_shot_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable>
    %closed = tle.pipe.reader_wait %a[%c0, %false] {async_task_id = array<i32: 1>, capacity = 1 : i32, pipe_name = "one_shot_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable>
    scf.if %closed {
    }
    tle.pipe.reader_release %a[%c0] {async_task_id = array<i32: 1>, capacity = 1 : i32, pipe_name = "one_shot_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}
