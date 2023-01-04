#include <stdexcept>

#include "ppmatrix.hpp"
#include "pputils.hpp"
#include "../util/helper.hpp"

#include "fmt/core.h"


__host__ void ppgemm_f32(
    cublasHandle_t handle,
    char transA,
    char transB,
    f32 alpha,
    PPMatrix<f32>& A,
    PPMatrix<f32>& B,
    f32 beta,
    PPMatrix<f32>& C
) {
    fmt::print("/!\\ not implemented\n");
    throw std::exception();
}
 
static void handle_err(cudaError_t val, int line) {
    if (__builtin_expect(val != cudaSuccess, 0)) {
        fmt::print("CUDA error at line {}: {}\n", line, cudaGetErrorString(cudaGetLastError()));
        throw std::exception();
    }
}

__host__ void ppgemm_f64(
    cublasHandle_t handle,
    char transA,
    char transB,
    f64 alpha,
    PPMatrix<f64>& A,
    PPMatrix<f64>& B,
    f64 beta,
    PPMatrix<f64>& C
) {
    assert(A.rows == C.rows);
    assert(B.cols == C.cols);
    assert(A.cols == B.rows);

    int m = static_cast<int>((transA == 'N') ? A.rows : A.cols);
    int n = static_cast<int>((transB == 'N') ? B.cols : B.rows);
    int k = static_cast<int>((transA == 'N') ? A.cols : A.rows);

    // ####################
    // STEP 0: preparation
    // ####################

    // alloc CPU buffers
    float *A_h, *A_l, *B_h, *B_l, *C_h, *C_l;
    
    handle_err(cudaMallocHost(&A_h, A.rows * A.cols * sizeof(float)), __LINE__);
    handle_err(cudaMallocHost(&A_l, A.rows * A.cols * sizeof(float)), __LINE__);

    handle_err(cudaMallocHost(&B_h, B.rows * B.cols * sizeof(float)), __LINE__);
    handle_err(cudaMallocHost(&B_l, B.rows * B.cols * sizeof(float)), __LINE__);

    handle_err(cudaMallocHost(&C_h, C.rows * C.cols * sizeof(float)), __LINE__);
    handle_err(cudaMallocHost(&C_l, C.rows * C.cols * sizeof(float)), __LINE__);

    // alloc GPU buffers
    float *dA_h, *dA_l, *dB_h, *dB_l, *dC_h, *dC_l;
    double *dA_dgemm, *dB_dgemm, *dRes_dgemm;

    handle_err(cudaMalloc(&dA_h, A.rows * A.cols * sizeof(float)), __LINE__);
    handle_err(cudaMalloc(&dA_l, A.rows * A.cols * sizeof(float)), __LINE__);

    handle_err(cudaMalloc(&dB_h, B.rows * B.cols * sizeof(float)), __LINE__);
    handle_err(cudaMalloc(&dB_l, B.rows * B.cols * sizeof(float)), __LINE__);

    handle_err(cudaMalloc(&dC_h, C.rows * C.cols * sizeof(float)), __LINE__);
    handle_err(cudaMalloc(&dC_l, C.rows * C.cols * sizeof(float)), __LINE__);

    handle_err(cudaMalloc(&dA_dgemm, A.rows * A.cols * sizeof(double)), __LINE__);
    handle_err(cudaMalloc(&dB_dgemm, B.rows * B.cols * sizeof(double)), __LINE__);
    handle_err(cudaMalloc(&dRes_dgemm, C.rows * C.cols * sizeof(double)), __LINE__);

    // use the default stream so that no sync are needed before / after a call to this function
    cudaStream_t s0 = 0;

    // register streams
    // streams are created as cudaStreamNonBlocking, meaning they do not
    // implicitly sync with the default stream
    cudaStream_t s1, s2, s3;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking);

    // register events
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // ###########################################
    // STEP 1: precision-decompose and send A & B
    // ###########################################

    // No need to use events here, communications from/to the device are serialized

    // decompose A on the host
    de_f64f32(A.ptr, A_h, A_l, A.rows, A.cols, A.ld);

    
    // copy A_h to GPU
    cudaMemcpyAsync(dA_h, A_h, A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice, s0);

    // decompose B on the host (we're already copying A_h to GPU, so we can do this in parallel)
    de_f64f32(B.ptr, B_h, B_l, B.rows, B.cols, B.ld);

    // copy B_h to GPU
    cudaMemcpyAsync(dB_h, B_h, B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice, s2);

    // copy A_l to GPU
    cudaMemcpyAsync(dA_l, A_l, A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice, s1);

    // copy B_l to GPU
    cudaMemcpyAsync(dB_l, B_l, B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice, s3);
    cudaEventRecord(e1, s3);

    // ###################################################
    // STEP 2: convert dA_h and dB_h to double
    // ###################################################

    // launch the kernel in blocks of 256 threads
    // we removed the LD when we decomposed the initial matrices
    // from now on, we can treat the buffers as matrices where ld = rows
    f32tof64_flat<<<ceilDiv(A.rows * A.cols, 256U), 256, 0, s0>>>(dA_h, dA_dgemm, A.rows * A.cols);

    f32tof64_flat<<<ceilDiv(B.rows * B.cols, 256U), 256, 0, s2>>>(dB_h, dB_dgemm, B.rows * B.cols);

    // notify that the conversion is done for dB_h
    cudaEventRecord(e0, s2);


    // ###################################################
    // STEP 3: dgemm on dA and dB
    // ###################################################

    // wait for s2 to finish converting dB_h into dB_dgemm
    cudaStreamWaitEvent(s0, e0, 0);

    // perform the dgemm (dRes_dgemm = dA_dgemm * dB_dgemm) on S0
    cublasSetStream(handle, s0);

    cublasDgemm(
        handle,
        convertToCublas(transA), convertToCublas(transB),
        m, n, k,
        &alpha,
        dA_dgemm, A.rows,
        dB_dgemm, B.rows,
        &beta,
        dRes_dgemm, C.rows
    );
    
    // notify that the dgemm on s0 is done
    cudaEventRecord(e0, s0);

    // ###################################################
    // STEP 4: decompose dRes_dgemm into dC_h and dC_l
    // ###################################################

    // perform the decomposition immediatly on s0 because
    // it is the stream which performed the dgemm
    extractf32high_flat<<<ceilDiv(C.rows * C.cols, 256U), 256, 0, s0>>>(dRes_dgemm, dC_h, C.rows * C.cols);

    // wait for s0 to finish dgemm and perform the decomposition on s2
    cudaStreamWaitEvent(s2, e0, 0);
    extractf32low_flat<<<ceilDiv(C.rows * C.cols, 256U), 256, 0, s2>>>(dRes_dgemm, dC_l, C.rows * C.cols);
    cudaEventRecord(e0, s2); // s2 finished the decomposition

    // ###################################################
    // STEP 5: send back dC_h to the host
    // ###################################################

    cudaMemcpyAsync(C_h, dC_h, C.rows * C.cols * sizeof(float), cudaMemcpyDeviceToHost, s0);

    // ###################################################
    // STEP 6: accumulate sgemm rounds
    // ###################################################

    float sgemm_alpha = 1.0f;
    float sgemm_beta = 1.0f;

    // wait for s2 to finish the dC_l decomposition
    // and perform the sgemm dC_l =  dA_h * dB_l + dC_l
    cudaStreamWaitEvent(s1, e0, 0);
    cublasSetStream(handle, s1);
    cublasSgemm(
        handle,
        convertToCublas(transA), convertToCublas(transB),
        m, n, k,
        &sgemm_alpha,
        dA_h, A.rows,
        dB_l, B.rows,
        &sgemm_beta,
        dC_l, C.rows
    );

    // wait for s4 to finish the upload of dC_h
    // and perform the sgemm dC_l = dA_l * dB_h + dC_l
    cudaStreamWaitEvent(s1, e1, 0);
    cublasSgemm(
        handle,
        convertToCublas(transA), convertToCublas(transB),
        m, n, k,
        &sgemm_alpha,
        dA_l, A.rows,
        dB_h, B.rows,
        &sgemm_beta,
        dC_l, C.rows
    );

    // ###################################################
    // STEP 7: send the result back to the host
    // ###################################################
    
    cudaMemcpyAsync(C_l, dC_l, C.rows * C.cols * sizeof(float), cudaMemcpyDeviceToHost, s1);

    // ###################################################
    // STEP 8 wait for everything to finish
    // ###################################################

    cudaStreamSynchronize(s0);
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s3);
    cudaStreamSynchronize(s2);

    // ###################################################
    // STEP 9: cleanup
    // ###################################################

    cudaStreamDestroy(s0);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaStreamDestroy(s3);

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    // ###################################################
    // STEP 10: precision-recompose C
    // ###################################################
    re_f64f32(C_h, C_l, C.ptr, C.rows, C.cols, C.ld);

    cudaFree(dA_h);
    cudaFree(dA_l);
    cudaFree(dA_dgemm);
    cudaFree(dB_h);
    cudaFree(dB_l);
    cudaFree(dB_dgemm);
    cudaFree(dC_h);
    cudaFree(dC_l);
    cudaFree(dRes_dgemm);

    cudaFreeHost(A_h);
    cudaFreeHost(A_l);
    cudaFreeHost(B_h);
    cudaFreeHost(C_h);
    cudaFreeHost(C_l);
}
