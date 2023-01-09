# StarPU GEMM and NDIM example
This repository is an example of how to use the [StarPU runtime](https://gitlab.inria.fr/starpu/starpu) on GEMM and tensor addition (works with and without CUDA)

### Dependencies
Depends on a BLAS package, CUDA, cublas, MPI and StarPU 1.3
Tested as working using
* openblas 0.3.21
* CUDA 11.7
* cublas 11.0
* starpu 1.3.99
* openmpi
* [fmt]{https://github.com/fmtlib/fmt}

### Build
To build and run the experimental code
```
cmake -B build [-DENABLE_CUDA=ON]
cd build
cmake --build .
./gemm [exp] [k_min] [k_max] [bs_min] [bs_max]
```

### Executing with Guix
This is integrated as a package in the [Guix HPC](https://gitlab.inria.fr/guix-hpc/guix-hpc) and [Guix HPC non-free](https://gitlab.inria.fr/guix-hpc/guix-hpc-non-free) repositories.

Before executing the example, make sure to set `OMP_NUM_THREADS=1`, to make openblas use a single thread

<!---
To run the gemm example with CUDA run the following example command in your terminal (note : your CUDA driver must be more recent than the one in Guix)
```
LD_PRELOAD=/usr/lib64/libcuda.so OMPI_MCA_btl=^openib OMPI_MCA_osc=^ucx OMPI_MCA_pml=^ucx guix shell --pure --preserve="^OMPI_MCA|^LD_PRELOAD|^SLURM|^STARPU|^OMP" slurm@22 starpu-example-cppgemm-cuda --with-branch=starpu-example-cppgemm-cuda=main --with-branch=starpu-cuda=master openssh -- srun -l gemm 3 3 3 2 2
```
--->

To run the gemm example without CUDA run the following example command in your terminal
```
OMPI_MCA_btl=^openib OMPI_MCA_osc=^ucx OMPI_MCA_pml=^ucx guix shell --pure --preserve="^OMPI_MCA|^SLURM|^STARPU|^OMP" openssh slurm@22 starpu-example-cppgemm --with-branch=starpu-example-cppgemm=main --with-branch=starpu=master -- srun -l gemm [exp] [k_min] [k_max] [bs_min] [bs_max]
```
