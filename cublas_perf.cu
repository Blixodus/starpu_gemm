#include "cublas_v2.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include "cublas_perf.hpp"

void cublas_perf_test(int m, int n, int k, bool pin, std::ofstream& resultFile) {
  std::cerr << "================= BEGIN CUBLAS PERF TEST ===============" << std::endl;
  
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  
  size_t size_A = m*k*sizeof(float);
  size_t size_B = k*n*sizeof(float);
  size_t size_C = m*n*sizeof(float);

  
  float * h_A, * h_B, * h_C;
  if(pin) {
    cudaStat = cudaMallocHost((void**)&h_A, size_A);
    if (cudaStat != cudaSuccess) {
      printf ("host memory allocation failed");
    }
    cudaStat = cudaMallocHost((void**)&h_B, size_B);
    if (cudaStat != cudaSuccess) {
      printf ("host memory allocation failed");
    }
    cudaStat = cudaMallocHost((void**)&h_C, size_C);
    if (cudaStat != cudaSuccess) {
      printf ("host memory allocation failed");
    }
  } else {
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);
  }

  //for(int i = 0; i < size_A/sizeof(float); i++) { h_A[i] = 1; }
  //for(int i = 0; i < size_B/sizeof(float); i++) { h_B[i] = 1; }
  //for(int i = 0; i < size_C/sizeof(float); i++) { h_C[i] = 0; }
    
  float * d_A, * d_B, * d_C;
  cudaStat = cudaMalloc(&d_A, size_A);
  if (cudaStat != cudaSuccess) {
    printf ("device memory allocation failed");
  }
  cudaStat = cudaMalloc(&d_B, size_B);
  if (cudaStat != cudaSuccess) {
    printf ("device memory allocation failed");
  }
  cudaStat = cudaMalloc(&d_C, size_C);
  if (cudaStat != cudaSuccess) {
    printf ("device memory allocation failed");
  }

  auto startAB = std::chrono::high_resolution_clock::now();
  cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
  std::chrono::duration<double> timeAB = std::chrono::high_resolution_clock::now() - startAB;
  std::cerr << "Transfer time : " << timeAB.count() << "s" << std::endl;
  std::cerr << "Performance : " << (size_A + size_B) / timeAB.count() / 1e9 << "GB/s" << std::endl;
  
  stat = cublasCreate(&handle);
  if(stat != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS initialization failed\n");
  }
  auto startBLAS = std::chrono::high_resolution_clock::now();
  float alpha = 1.0, beta = 0.0;
  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
  cudaDeviceSynchronize();
  std::chrono::duration<double> timeBLAS = std::chrono::high_resolution_clock::now() - startBLAS;
  std::cerr << "BLAS time : " << timeBLAS.count() << "s" << std::endl;
  std::cerr << "Performance : " << 2L * m * n * k / timeBLAS.count() / 1e9 << "Gflop/s" << std::endl;
  if(stat != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS GEMM failed\n");
  }
  cublasDestroy(handle);

  auto startC = std::chrono::high_resolution_clock::now();
  cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
  std::chrono::duration<double> timeC = std::chrono::high_resolution_clock::now() - startC;
  std::cerr << "Transfer time : " << timeC.count() << "s" << std::endl;
  std::cerr << "Performance : " << (size_C) / timeC.count() / 1e9 << "GB/s" << std::endl;

  
  std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - startAB;
  std::cerr << "Time : " << time.count() << "s" << std::endl;
  std::cerr << "Performance : " << 2L * m * n * k / time.count() / 1e9 << "Gflop/s" << std::endl;
  
  resultFile << 0 << ";" << 0 << ";" << m << ";" << n << ";" << k << ";" << 0 << ";" << 2L * m * n * k / time.count() / 1e12 << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_C);

  std::cerr << "================= END CUBLAS PERF TEST ===============" << std::endl;
}
