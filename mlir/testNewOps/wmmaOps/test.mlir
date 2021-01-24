module attributes {gpu.container_module}  {
  func @main() {
    %0 = alloc() : memref<2x6xi32>
    %1 = alloc() : memref<2xi32>
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    %c2_i32 = constant 2 : i32
    %c4_i32 = constant 4 : i32
    %c8_i32 = constant 8 : i32
    %c16_i32 = constant 16 : i32
    %c3_i32 = constant 3 : i32
    %c6_i32 = constant 6 : i32
    %c7_i32 = constant 7 : i32
    %c10_i32 = constant 10 : i32
    %c11_i32 = constant 11 : i32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %c4 = constant 4 : index
    %c5 = constant 5 : index
    %c6 = constant 6 : index
    %2 = memref_cast %0 : memref<2x6xi32> to memref<*xi32>
    gpu.host_register %2 : memref<*xi32>
    %3 = memref_cast %1 : memref<2xi32> to memref<*xi32>
    gpu.host_register %3 : memref<*xi32>
    store %c0_i32, %0[%c0, %c0] : memref<2x6xi32>
    store %c1_i32, %0[%c0, %c1] : memref<2x6xi32>
    store %c2_i32, %0[%c0, %c2] : memref<2x6xi32>
    store %c4_i32, %0[%c0, %c3] : memref<2x6xi32>
    store %c8_i32, %0[%c0, %c4] : memref<2x6xi32>
    store %c16_i32, %0[%c0, %c5] : memref<2x6xi32>
    store %c2_i32, %0[%c1, %c0] : memref<2x6xi32>
    store %c3_i32, %0[%c1, %c1] : memref<2x6xi32>
    store %c6_i32, %0[%c1, %c2] : memref<2x6xi32>
    store %c7_i32, %0[%c1, %c3] : memref<2x6xi32>
    store %c10_i32, %0[%c1, %c4] : memref<2x6xi32>
    store %c11_i32, %0[%c1, %c5] : memref<2x6xi32>
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c2, %c1, %c1) threads in (%c6, %c1, %c1) args(%0 : memref<2x6xi32>, %1 : memref<2xi32>)
    call @print_memref_i32(%3) : (memref<*xi32>) -> ()
    return
  }
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<2x6xi32>, %arg1: memref<2xi32>) private(%pv : memref<16xf32, 5>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.block_id"() {dimension = "y"} : () -> index
      %2 = "gpu.block_id"() {dimension = "z"} : () -> index
      %3 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %4 = "gpu.thread_id"() {dimension = "y"} : () -> index
      %5 = "gpu.thread_id"() {dimension = "z"} : () -> index
      %6 = "gpu.grid_dim"() {dimension = "x"} : () -> index
      %7 = "gpu.grid_dim"() {dimension = "y"} : () -> index
      %8 = "gpu.grid_dim"() {dimension = "z"} : () -> index
      %9 = "gpu.block_dim"() {dimension = "x"} : () -> index
      %10 = "gpu.block_dim"() {dimension = "y"} : () -> index
      %11 = "gpu.block_dim"() {dimension = "z"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0
      %12 = load %arg0[%0, %3] : memref<2x6xi32>
      %13 = "gpu.all_reduce"(%12) ( {
      }) {op = "max"} : (i32) -> i32
      store %13, %arg1[%0] : memref<2xi32>
      gpu.return
    }
  }
  func private @print_memref_i32(memref<*xi32>)
}
