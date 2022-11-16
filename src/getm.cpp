#include <starpu.h>
#include <starpu_mpi.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <fstream>

#include "tensor/tensor.hpp"
#include "util/helper.hpp"


void test_tensor() {
	Tensor<double> A({4, 4, 4, 4}, 2);
	Tensor<double> B({4, 4, 4, 4}, 2);
	Tensor<double> C({4, 4, 4, 4}, 2);
	A.fill(8);
	B.fill(1);
	C.fill(0);
	Tensor<double>::add(A, B, C);
	C.assertEq(9);
}

int main(int argc, char ** argv) {
	int err = starpu_init(NULL);
	if(err) { throw std::exception(); }
	err = starpu_mpi_init(&argc, &argv, 1);
	if (err) { throw std::exception(); }
	starpu_cublas_init();

	test_tensor();
	
	starpu_cublas_shutdown();
	starpu_mpi_shutdown();
	starpu_shutdown();
	return 0;
}
