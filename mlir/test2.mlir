func @matmul(){
  %m3 = alloc() : memref<8x8xvector<16xf16>>
  %c0 = constant 0 : index
  %x = load %m3[%c0, %c0] : memref<8x8xvector<16xf16>> 
  store %x, %m3[%c0, %c0] : memref<8x8xvector<16xf16>>
  
  return
}
