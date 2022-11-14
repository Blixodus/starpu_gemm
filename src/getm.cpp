#include <starpu.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <fstream>

#include "tensor/tensor_fill_func.hpp"
#include "tensor/tensor_add_func.hpp"
#include "tensor/tensor.hpp"

void test_tensor() {
  std::vector<unsigned int> dim_size = {20, 20, 20, 20};
  unsigned int block_size = 10;
  Tensor<double> A(dim_size, block_size);
  //Tensor<double> B(ndim, dim_size, tile_size);
  A.fill(8);
  //B.fill(1);
  //auto C = A + B;
}

int main(int argc, char ** argv) {
  int err = starpu_init(NULL);
  if(err) { throw std::exception(); }
  starpu_cublas_init();

  test_tensor();
  
  starpu_cublas_shutdown();
  starpu_shutdown();
  return 0;
}
