# starpu_gemm
Test for GEMM using STARPU runtime


### Build
To build and run the experimental code
```
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .
$ ./gemm [exp] [k_min] [k_max] [bs_min] [bs_max]
```