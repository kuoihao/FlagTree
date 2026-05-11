// RUN: triton-opt %s -split-input-file -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-pipeline -canonicalize | FileCheck %s

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @defer_wait_for_same_iteration_wgmma_c_chain
  tt.func @defer_wait_for_same_iteration_wgmma_c_chain(
      %a0: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b0: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %a1: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b1: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %b0_view = tle.memdesc_wgmma_view %b0 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem>
    %b1_view = tle.memdesc_wgmma_view %b1 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem>
    scf.for %iv = %c0 to %c8 step %c1 {
      // CHECK: %[[DOT0:.+]] = ttng.warp_group_dot
      %dot0 = ttng.warp_group_dot %a0, %b0_view, %zero {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: %[[DOT1:.+]] = ttng.warp_group_dot {{.*}}, %[[DOT0]]
      %dot1 = ttng.warp_group_dot %a1, %b1_view, %dot0 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: %[[WAIT:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT1]]
      // CHECK-SAME: {pendings = 0 : i32}
      // CHECK: tt.store %{{.*}}, %[[WAIT]]#0
      tt.store %out, %dot1 : tensor<64x64x!tt.ptr<f32>, #mma>
    }
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @lowered_release_forces_wait_before_lifetime_boundary
  tt.func @lowered_release_forces_wait_before_lifetime_boundary(
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %barrier: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
      %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %zero) -> (tensor<64x64xf32, #mma>) {
      // CHECK: %[[DOT:.+]] = ttng.warp_group_dot
      %dot = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: %[[WAIT:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT]]{{.*}} {pendings = 0 : i32}
      // CHECK-NEXT: ttng.arrive_barrier
      ttng.arrive_barrier %barrier, 128 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
      scf.yield %dot : tensor<64x64xf32, #mma>
    }
    tt.store %out, %res : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @release_forces_wait_before_lifetime_boundary
  tt.func @release_forces_wait_before_lifetime_boundary(
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %out: tensor<64x64x!tt.ptr<f32>, #mma>,
      %idx: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %token = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %zero) -> (tensor<64x64xf32, #mma>) {
      // CHECK: %[[DOT:.+]] = ttng.warp_group_dot
      %dot = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: %[[WAIT:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT]]{{.*}} {pendings = 0 : i32}
      // CHECK-NEXT: nvws.consumer_release
      nvws.consumer_release %token, %idx {release_count = 128 : i32} : tensor<2x!nvws.token>, i32
      scf.yield %dot : tensor<64x64xf32, #mma>
    }
    tt.store %out, %res : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}
