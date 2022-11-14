#include <starpu.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <fstream>

#include "tensor_fill_func.hpp"
#include "tensor_add_func.hpp"
#include "tensor.hpp"

void test_tensor() {
  constexpr size_t ndim = 4;
  std::vector<unsigned int> dim_size = {20, 20, 20, 20};
  std::vector<unsigned int> tile_size = {10, 2, 2, 2};
  Tensor<double> A(ndim, dim_size, tile_size);
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
