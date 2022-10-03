# starpu_gemm
Test for GEMM using STARPU runtime

### Dependencies
Depends on a BLAS package, CUDA, cublas and StarPU 1.3
Tested as working using
* openblas 0.3.21
* CUDA 11.7
* cublas 11.0
* starpu 1.3.99

### Build
To build and run the experimental code
```
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .
$ ./gemm [exp] [k_min] [k_max] [bs_min] [bs_max]
```