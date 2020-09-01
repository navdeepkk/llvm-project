// RUN: mlir-opt %s -split-input-file -affine-loop-tile="tile-using-parameters" | FileCheck %s

// -----

// CHECK-DAG: [[LBI:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: [[UBI0:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0 + s0, 256)>
// CHECK-DAG: [[UBI1:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0 + s0, 512)>
// CHECK-DAG: [[UBI2:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0 + s0, 1024)>
// CHECK-DAG: [[UBO0:#map[0-9]+]] = affine_map<()[s0] -> (256 ceildiv s0)>
// CHECK-DAG: [[UBO1:#map[0-9]+]] = affine_map<()[s0] -> (512 ceildiv s0)>
// CHECK-DAG: [[UBO2:#map[0-9]+]] = affine_map<()[s0] -> (1024 ceildiv s0)>

// CHECK: func @loop_tiling([[ARG0:%arg[0-9]+]]: index, [[ARG1:%arg[0-9]+]]: index, [[ARG2:%arg[0-9]+]]: index)
// CHECK-NEXT:   affine.for [[ARG3:%arg[0-9]+]] = 0 to [[UBO0]](){{.*}}[[ARG0]]
// CHECK-NEXT:     affine.for [[ARG4:%arg[0-9]+]] = 0 to [[UBO1]](){{.*}}[[ARG1]]
// CHECK-NEXT:       affine.for [[ARG5:%arg[0-9]+]] = 0 to [[UBO2]](){{.*}}[[ARG2]]
// CHECK-NEXT:         affine.for %[[I:.*]] = [[LBI]]{{.*}}[[ARG3]]{{.*}}[[ARG0]]{{.*}} to min [[UBI0]]{{.*}}[[ARG3]]{{.*}}[[ARG0]]
// CHECK-NEXT:          affine.for %[[J:.*]] = [[LBI]]{{.*}}[[ARG4]]{{.*}}[[ARG1]]{{.*}} to min [[UBI1]]{{.*}}[[ARG4]]{{.*}}[[ARG1]]
// CHECK-NEXT:            affine.for %[[K:.*]] = [[LBI]]{{.*}}[[ARG5]]{{.*}}[[ARG2]]{{.*}} to min [[UBI2]]{{.*}}[[ARG5]]{{.*}}[[ARG2]]
// CHECK-NEXT:              "test.foo"(%[[I]], %[[J]], %[[K]])

func @loop_tiling(%t0 : index, %t1 : index, %t2 : index) {
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 512 {
      affine.for %k = 0 to 1024 {
        "test.foo"(%i, %j, %k) : (index, index, index) -> ()
      }
    }
  }
  return
}

// -----

// CHECK-DAG: [[LBI:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: [[UBI0:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0 + s0, 50)>
// CHECK-DAG: [[UBO0:#map[0-9]+]] = affine_map<()[s0] -> (50 ceildiv s0)>

// CHECK: func @loop_tiling_1([[ARG0:%arg[0-9]+]]: index)
// CHECK-NEXT:   affine.for [[ARG1:%arg[0-9]+]] = 0 to [[UBO0]](){{.*}}[[ARG0]]
// CHECK-NEXT:     affine.for %[[I:.*]] = [[LBI]]{{.*}}[[ARG1]]{{.*}}[[ARG0]]{{.*}} to min [[UBI0]]{{.*}}[[ARG1]]{{.*}}[[ARG0]]
// CHECK-NEXT:       "test.bar"(%[[I]], %[[I]])

func @loop_tiling_1(%t3 : index){
  affine.for %x = 0 to 50 {
    "test.bar"(%x, %x) : (index, index) -> ()
  }
  return
}

// -----

// CHECK-DAG: [[LBI0:#map[0-9]+]] = affine_map<(d0)[s0, s1] -> (d0 * s1)>
// CHECK-DAG: [[UBI0:#map[0-9]+]] = affine_map<(d0)[s0, s1, s2] -> (d0 * s2 + s2, s0, 4096 floordiv s1)>
// CHECK-DAG: [[LBO0:#map[0-9]+]] = affine_map<()[s0] -> (0, s0)>
// CHECK-DAG: [[UBO0:#map[0-9]+]] = affine_map<()[s0, s1, s2] -> (s0 ceildiv s2, (4096 floordiv s1) ceildiv s2)>

#lb = affine_map<()[s0] -> (0, s0)>
#ub = affine_map<()[s0, s1] -> (s0, 4096 floordiv s1)>
// CHECK: func @loop_max_min_bound([[ARG0:%arg[0-9]+]]: index, %{{.*}}: memref<?xi32>, %{{.*}}: index, %{{.*}}: index)
func @loop_max_min_bound(%t5 : index, %A : memref<? x i32>, %L : index, %U : index) {
  %c0 = constant 0 : index
  %M = dim %A, %c0 : memref<? x i32>
  affine.for %i = max #lb()[%L] to min #ub()[%M, %U] {
    addi %i, %i : index
  }
  // CHECK:  affine.for [[ARG1:%arg[0-9]+]] = max [[LBO0]]()[%{{.*}}] to min [[UBO0]]()[%{{.*}}, %{{.*}}, [[ARG0]]]
  // CHECK-NEXT:    affine.for %[[I:.*]] = [[LBI0]]([[ARG1]])[{{.*}}, [[ARG0]]] to min [[UBI0]]({{.*}})[{{.*}}, {{.*}}, [[ARG0]]] 
  // CHECK-NEXT:      addi %[[I]], %[[I]]
  return
}

// -----

// CHECK-DAG: [[LBI:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: [[UBI0:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0 + s0, 256)>
// CHECK-DAG: [[UBI1:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0 + s0, 250)>
// CHECK-DAG: [[UBO0:#map[0-9]+]] = affine_map<()[s0] -> (256 ceildiv s0)>
// CHECK-DAG: [[UBO1:#map[0-9]+]] = affine_map<()[s0] -> (250 ceildiv s0)>

// CHECK: func @simple_matmul([[ARG0:%arg[0-9]+]]: index, [[ARG1:%arg[0-9]+]]: index, [[ARG2:%arg[0-9]+]]: index{{.*}})
// CHECK-NEXT:   affine.for [[ARG3:%arg[0-9]+]] = 0 to [[UBO0]](){{.*}}[[ARG0]]{{.*}}
// CHECK-NEXT:     affine.for [[ARG4:%arg[0-9]+]] = 0 to [[UBO0]](){{.*}}[[ARG1]]{{.*}}
// CHECK-NEXT:       affine.for [[ARG5:%arg[0-9]+]] = 0 to [[UBO1]](){{.*}}[[ARG2]]{{.*}}
// CHECK-NEXT:         affine.for %[[I:.*]] = [[LBI]]{{.*}}[[ARG3]]{{.*}}[[ARG0]]{{.*}} to min [[UBI0]]{{.*}}[[ARG3]]{{.*}}[[ARG0]]{{.*}}
// CHECK-NEXT:          affine.for %[[J:.*]] = [[LBI]]{{.*}}[[ARG4]]{{.*}}[[ARG1]]{{.*}} to min [[UBI0]]{{.*}}[[ARG4]]{{.*}}[[ARG1]]{{.*}}
// CHECK-NEXT:            affine.for %[[K:.*]] = [[LBI]]{{.*}}[[ARG5]]{{.*}}[[ARG2]]{{.*}} to min [[UBI1]]{{.*}}[[ARG5]]{{.*}}[[ARG2]]{{.*}}
// CHECK-NEXT:                 affine.load %{{.*}}[%[[I]], %[[K]]]
// CHECK-NEXT:                 affine.load %{{.*}}[%[[K]], %[[J]]]
// CHECK-NEXT:                 affine.load %{{.*}}[%[[I]], %[[J]]]
// CHECK-NEXT:                 mulf %{{.*}}
// CHECK-NEXT:                 addf %{{.*}}
// CHECK-NEXT:                 affine.store %{{.*}}[%[[I]], %[[J]]]
func @simple_matmul(%t6 : index, %t7 : index, %t8 : index, %arg0: memref<256x256xvector<64xf32>>, %arg1: memref<256x256xvector<64xf32>>, %arg2: memref<256x256xvector<64xf32>>) -> memref<256x256xvector<64xf32>> {
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 256 {
      affine.for %k = 0 to 250 {
        %l = affine.load %arg0[%i, %k] : memref<256x256xvector<64xf32>>
        %r = affine.load %arg1[%k, %j] : memref<256x256xvector<64xf32>>
        %o = affine.load %arg2[%i, %j] : memref<256x256xvector<64xf32>>
        %m = mulf %l, %r : vector<64xf32>
        %a = addf %o, %m : vector<64xf32>
        affine.store %a, %arg2[%i, %j] : memref<256x256xvector<64xf32>>
      }
    }
  }
  return %arg2 : memref<256x256xvector<64xf32>>
}

// -----

//CHECK-DAG: [[LBI0:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0)>
//CHECK-DAG: [[UBI0:#map[0-9]+]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s1, s0)>
//CHECK-DAG: [[UBO0:#map[0-9]+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>

// CHECK: func @tile_with_symbolic_loop_upper_bounds([[ARG0:%arg[0-9]+]]: index, [[ARG1:%arg[0-9]+]]: index{{.*}}){{.*}}
// CHECK:   affine.for [[ARG2:%arg[0-9]+]] = 0 to [[UBO0]](){{.*}}[[ARG0]]{{.*}}
// CHECK-NEXT:     affine.for [[ARG3:%arg[0-9]+]] = 0 to [[UBO0]](){{.*}}[[ARG1]]{{.*}}
// CHECK-NEXT:       affine.for %[[I:.*]] = [[LBI0]]{{.*}}[[ARG2]]{{.*}}[[ARG0]]{{.*}} to min [[UBI0]]{{.*}}[[ARG2]]{{.*}}[[ARG0]]{{.*}}
// CHECK-NEXT:         affine.for %[[J:.*]] = [[LBI0]]{{.*}}[[ARG3]]{{.*}}[[ARG1]]{{.*}} to min [[UBI0]]{{.*}}[[ARG3]]{{.*}}[[ARG1]]{{.*}}
// CHECK-NEXT:          affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
// CHECK-NEXT:          affine.for %{{.*}} = 0 to %{{.*}} {
// CHECK-NEXT:            affine.load
// CHECK-NEXT:            affine.load
// CHECK-NEXT:            mulf
// CHECK-NEXT:            affine.load
// CHECK-NEXT:            addf
// CHECK-NEXT:            affine.store
func @tile_with_symbolic_loop_upper_bounds(%t9 : index, %t10: index, %arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  %cst = constant 0.000000e+00 : f32
  %c0 = constant 0 : index
  %0 = dim %arg0, %c0 : memref<?x?xf32>
  affine.for %i0 = 0 to %0 {
    affine.for %i1 = 0 to %0 {
      affine.store %cst, %arg2[%i0, %i1] : memref<?x?xf32>
      affine.for %i2 = 0 to %0 {
        %1 = affine.load %arg0[%i0, %i2] : memref<?x?xf32>
        %2 = affine.load %arg1[%i2, %i1] : memref<?x?xf32>
        %3 = mulf %1, %2 : f32
        %4 = affine.load %arg2[%i0, %i1] : memref<?x?xf32>
        %5 = addf %4, %3 : f32
        affine.store %5, %arg2[%i0, %i1] : memref<?x?xf32>
      }
    }
  }
  return
}

// -----

// CHECK-DAG: [[LBI0:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: [[UBI0:#map[0-9]+]] = affine_map<(d0)[s0, s1, s2] -> (d0 * s2 + s2, s0 + s1)>
// CHECK-DAG: [[UBO0:#map[0-9]+]] = affine_map<()[s0, s1, s2] -> ((s0 + s1) ceildiv s2)>
// CHECK: func @tile_with_loop_upper_bounds_in_two_symbols([[ARG0:%arg[0-9]+]]: index{{.*}}){{.*}}

func @tile_with_loop_upper_bounds_in_two_symbols(%t11 : index, %arg0: memref<?xf32>, %limit: index) {
  %c0 = constant 0 : index
  %dim0 = dim %arg0, %c0 : memref<?xf32>
  affine.for %i0 = 0 to affine_map<()[s0, s1] -> (s0 + s1)> ()[%dim0, %limit] {
    %v0 = affine.load %arg0[%i0] : memref<?xf32>
  }
  // CHECK:  affine.for [[ARG1:%arg[0-9]+]] = 0 to [[UBO0]]()[%{{.*}}, %{{.*}}, [[ARG0]]]
  // CHECK-NEXT:    affine.for %[[I:.*]] = [[LBI0]]([[ARG1]]){{.*}}[[ARG0]]{{.*}} to min [[UBI0]]([[ARG1]])[{{.*}}, {{.*}}, [[ARG0]]]
  // CHECK-NEXT:      affine.load %{{.*}}[%[[I]]]
  return
}

// -----

// CHECK-DAG: [[LBI0:#map[0-9]+]] = affine_map<(d0, d1)[s0] -> (d1 * s0)>
// CHECK-DAG: [[UBI1:#map[0-9]+]] = affine_map<(d0, d1)[s0] -> (d1 * s0 + s0, d0 + 4)>
// CHECK-DAG: [[UBI0:#map[0-9]+]] = affine_map<(d0, d1)[s0] -> (d1 * s0 + s0, d0 + 2)>
// CHECK-DAG: [[LBO0:#map[0-9]+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[UBO1:#map[0-9]+]] = affine_map<(d0)[s0] -> ((d0 + 4) ceildiv s0)>
// CHECK-DAG: [[UBO0:#map[0-9]+]] = affine_map<(d0)[s0] -> ((d0 + 2) ceildiv s0)>

// CHECK: func @tile_size_larger_than_trip_count_symbolic_bound([[ARG0:%arg[0-9]+]]: index, [[ARG1:%arg[0-9]+]]: index{{.*}}){{.*}}
// CHECK:      affine.for [[ARG2:%arg[0-9]+]] = [[LBO0]]({{.*}}) to [[UBO0]]({{.*}}){{.*}}[[ARG0]] 
// CHECK-NEXT:   affine.for [[ARG3:%arg[0-9]+]] =  [[LBO0]]({{.*}}) to [[UBO1]]({{.*}}){{.*}}[[ARG1]]
// CHECK-NEXT:     affine.for {{.*}} = [[LBI0]]({{.*}}, [[ARG2]]){{.*}}[[ARG0]]{{.*}} to min [[UBI0]]({{.*}}, [[ARG2]]){{.*}}[[ARG0]]{{.*}}
// CHECK-NEXT:       affine.for {{.*}} = [[LBI0]]({{.*}}, [[ARG3]]){{.*}}[[ARG1]]{{.*}} to min [[UBI1]]({{.*}}, [[ARG3]]){{.*}}[[ARG1]]
func @tile_size_larger_than_trip_count_symbolic_bound(%t12 : index, %t13 :index, %M: index, %N :  index) {
  affine.for %i = affine_map<(d0) -> (d0)>(%M) to affine_map<(d0) -> (d0 + 2)>(%M) {
    affine.for %j = affine_map<(d0) -> (d0)>(%N) to affine_map<(d0) -> (d0 + 4)>(%N) {
      "test.foo" () : () -> ()
    }
  }
  return
}
