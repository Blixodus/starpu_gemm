#include <starpu.h>
#include <starpu_mpi.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
#include <array>


#include "tensor/tensor.hpp"


int main(int argc, char** argv) {
    auto v = mk_tensor<float, 2>({2, 2}, 2);
}
