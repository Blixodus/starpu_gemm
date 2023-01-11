#include <stdexcept>
#include <fmt/core.h>
#include <chrono>
#include <random>

#include "ppmatrix.hpp"
#include "pputils.hpp"

#include "../util/helper.hpp"
#include "../util/lapackAPI.hpp"

// #include <cutlass/cutlass.h>

template <typename DataType>
void PPMatrix<DataType>::rndFill() {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<DataType> dist(0, 1000);

    for (u32 i = 0; i < cols; ++i) {
        for (u32 j = 0; j < rows; ++j) {
            ptr[i * ld + j] = dist(e2);
        }
    }
}

template <typename DataType>
void PPMatrix<DataType>::fill(DataType e) {
    for (u32 i = 0; i < cols; ++i) {
        for (u32 j = 0; j < rows; ++j) {
            ptr[i * ld + j] = e;
        }
    }
}

template <typename DataType>
void PPMatrix<DataType>::assertEq(DataType e) {
    for (u32 i = 0; i < cols; ++i) {
        for (u32 j = 0; j < rows; ++j) {
            if (std::abs(ptr[i * ld + j] - e) > 1e-6) {
                fmt::print("Assertion failed at ({}, {}): {} != {}\n", i, j, ptr[i * ld + j], e);
            }
        }
    }
}

PerfRecord ppgemm_f32(
    cublasHandle_t handle,
    char transA,
    char transB,
    f32 alpha,
    const PPMatrix<f32>& A,
    const PPMatrix<f32>& B,
    f32 beta,
    PPMatrix<f32>& C
) {
    assert(A.rows == C.rows);
    assert(B.cols == C.cols);
    assert(A.cols == B.rows);

    int m = checked_cast<int>((transA == 'N') ? A.rows : A.cols);
    int n = checked_cast<int>((transB == 'N') ? B.cols : B.rows);
    int k = checked_cast<int>((transA == 'N') ? A.cols : A.rows);

    PerfRecord perf;

    // ###########################################
    // STEP 0: preparation
    // ###########################################

    // alloc CPU buffers
    f16 *A_1, *A_2, *A_3, *B_1, *B_2, *B_3, *C_1, *C_2, *C_3;

    HANDLE_ERR(cudaMallocHost(&A_1, A.rows * A.cols * sizeof(f16)));
    HANDLE_ERR(cudaMallocHost(&A_2, A.rows * A.cols * sizeof(f16)));
    HANDLE_ERR(cudaMallocHost(&A_3, A.rows * A.cols * sizeof(f16)));

    HANDLE_ERR(cudaMallocHost(&B_1, B.rows * B.cols * sizeof(f16)));
    HANDLE_ERR(cudaMallocHost(&B_2, B.rows * B.cols * sizeof(f16)));
    HANDLE_ERR(cudaMallocHost(&B_3, B.rows * B.cols * sizeof(f16)));

    HANDLE_ERR(cudaMallocHost(&C_1, C.rows * C.cols * sizeof(f16)));
    HANDLE_ERR(cudaMallocHost(&C_2, C.rows * C.cols * sizeof(f16)));
    HANDLE_ERR(cudaMallocHost(&C_3, C.rows * C.cols * sizeof(f16)));

    // alloc GPU buffers
    f16 *dA_1, *dA_2, *dA_3, *dB_1, *dB_2, *dB_3, *dC_1, *dC_2, *dC_3;

    HANDLE_ERR(cudaMalloc(&dA_1, A.rows * A.cols * sizeof(f16)));
    HANDLE_ERR(cudaMalloc(&dA_2, A.rows * A.cols * sizeof(f16)));
    HANDLE_ERR(cudaMalloc(&dA_3, A.rows * A.cols * sizeof(f16)));

    HANDLE_ERR(cudaMalloc(&dB_1, B.rows * B.cols * sizeof(f16)));
    HANDLE_ERR(cudaMalloc(&dB_2, B.rows * B.cols * sizeof(f16)));
    HANDLE_ERR(cudaMalloc(&dB_3, B.rows * B.cols * sizeof(f16)));

    HANDLE_ERR(cudaMalloc(&dC_1, C.rows * C.cols * sizeof(f16)));
    HANDLE_ERR(cudaMalloc(&dC_2, C.rows * C.cols * sizeof(f16)));
    HANDLE_ERR(cudaMalloc(&dC_3, C.rows * C.cols * sizeof(f16)));

    // use the default stream so that no sync are needed before / after a call to this function
    cudaStream_t s0 = 0;

    // register streams
    // streams are created as cudaStreamNonBlocking, meaning they do not
    // implicitly sync with the default stream
    cudaStream_t s1, s2, s3, s4, s5;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s4, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s5, cudaStreamNonBlocking);

    // register events
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // ###########################################
    // STEP 1: precision-decompose and send A & B
    // ###########################################

    // decompose A and B on the host (don't start copying yet to get accurate perf numbers)
    de_f32f16(A.ptr, A_1, A_2, A_3, A.rows, A.cols, A.ld);
    de_f32f16(B.ptr, B_1, B_2, B_3, B.rows, B.cols, B.ld);

    auto start = std::chrono::high_resolution_clock::now();

    // copy A_1, B_1, C_1 to GPU
    cudaMemcpyAsync(dA_1, A_1, A.rows * A.cols * sizeof(f16), cudaMemcpyHostToDevice, s0);
    cudaMemcpyAsync(dB_1, B_1, B.rows * B.cols * sizeof(f16), cudaMemcpyHostToDevice, s1);
    cudaMemcpyAsync(dC_1, C_1, C.rows * C.cols * sizeof(f16), cudaMemcpyHostToDevice, s2);

    // copy A_2, B_2, C_2 to GPU
    cudaMemcpyAsync(dA_2, A_2, A.rows * A.cols * sizeof(f16), cudaMemcpyHostToDevice, s3);
    cudaMemcpyAsync(dB_2, B_2, B.rows * B.cols * sizeof(f16), cudaMemcpyHostToDevice, s4);
    cudaMemcpyAsync(dC_2, C_2, C.rows * C.cols * sizeof(f16), cudaMemcpyHostToDevice, s5);

    // ###################################################
    // STEP 2: convert dA_1, dB_1 and dC_1 to fp32
    // ###################################################

    // ###################################################
    // STEP 8 wait for everything to finish
    // ###################################################

    cudaStreamSynchronize(s0);
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);
    cudaStreamSynchronize(s3);
    cudaStreamSynchronize(s4);
    cudaStreamSynchronize(s5);

    perf.compute = std::chrono::high_resolution_clock::now() - start;

    // ###################################################
    // STEP 9: cleanup
    // ###################################################

    cudaStreamDestroy(s0);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaStreamDestroy(s3);
    cudaStreamDestroy(s4);
    cudaStreamDestroy(s5);

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    // ###################################################
    // STEP 10: precision-recompose C
    // ###################################################
    re_f32f16(C_1, C_2, C_3, C.ptr, C.rows, C.cols, C.ld);

    cudaFree(dA_1);
    cudaFree(dA_2);
    cudaFree(dA_3);
    cudaFree(dB_1);
    cudaFree(dB_2);
    cudaFree(dB_3);
    cudaFree(dC_1);
    cudaFree(dC_2);
    cudaFree(dC_3);

    cudaFreeHost(A_1);
    cudaFreeHost(A_2);
    cudaFreeHost(A_3);
    cudaFreeHost(B_1);
    cudaFreeHost(B_2);
    cudaFreeHost(B_3);
    cudaFreeHost(C_1);
    cudaFreeHost(C_2);
    cudaFreeHost(C_3);

    return perf;
}

PerfRecord ppgemm_f64(
    cublasHandle_t handle,
    char transA,
    char transB,
    f64 alpha,
    const PPMatrix<f64>& A,
    const PPMatrix<f64>& B,
    f64 beta,
    PPMatrix<f64>& C
) {
    assert(A.rows == C.rows);
    assert(B.cols == C.cols);
    assert(A.cols == B.rows);

    int m = checked_cast<int>((transA == 'N') ? A.rows : A.cols);
    int n = checked_cast<int>((transB == 'N') ? B.cols : B.rows);
    int k = checked_cast<int>((transA == 'N') ? A.cols : A.rows);

    PerfRecord perf;

    // ###########################################
    // STEP 0: preparation
    // ###########################################

    // alloc CPU buffers
    float *A_h, *A_l, *B_h, *B_l, *C_h, *C_l;
    
    HANDLE_ERR(cudaMallocHost(&A_h, A.rows * A.cols * sizeof(float)));
    HANDLE_ERR(cudaMallocHost(&A_l, A.rows * A.cols * sizeof(float)));

    HANDLE_ERR(cudaMallocHost(&B_h, B.rows * B.cols * sizeof(float)));
    HANDLE_ERR(cudaMallocHost(&B_l, B.rows * B.cols * sizeof(float)));

    HANDLE_ERR(cudaMallocHost(&C_h, C.rows * C.cols * sizeof(float)));
    HANDLE_ERR(cudaMallocHost(&C_l, C.rows * C.cols * sizeof(float)));

    // alloc GPU buffers
    float *dA_h, *dA_l, *dB_h, *dB_l, *dC_h, *dC_l;
    double *dA_dgemm, *dB_dgemm, *dRes_dgemm;

    HANDLE_ERR(cudaMalloc(&dA_h, A.rows * A.cols * sizeof(float)));
    HANDLE_ERR(cudaMalloc(&dA_l, A.rows * A.cols * sizeof(float)));

    HANDLE_ERR(cudaMalloc(&dB_h, B.rows * B.cols * sizeof(float)));
    HANDLE_ERR(cudaMalloc(&dB_l, B.rows * B.cols * sizeof(float)));

    HANDLE_ERR(cudaMalloc(&dC_h, C.rows * C.cols * sizeof(float)));
    HANDLE_ERR(cudaMalloc(&dC_l, C.rows * C.cols * sizeof(float)));

    HANDLE_ERR(cudaMalloc(&dA_dgemm, A.rows * A.cols * sizeof(double)));
    HANDLE_ERR(cudaMalloc(&dB_dgemm, B.rows * B.cols * sizeof(double)));
    HANDLE_ERR(cudaMalloc(&dRes_dgemm, C.rows * C.cols * sizeof(double)));

    // use the default stream so that no sync are needed before / after a call to this function
    cudaStream_t s0 = 0;

    // register streams
    // streams are created as cudaStreamNonBlocking, meaning they do not
    // implicitly sync with the default stream
    cudaStream_t s1, s2, s3;
    HANDLE_ERR(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    HANDLE_ERR(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    HANDLE_ERR(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    HANDLE_ERR(cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking));

    // register events
    cudaEvent_t e0, e1;
    HANDLE_ERR(cudaEventCreate(&e0));
    HANDLE_ERR(cudaEventCreate(&e1));

    // ###########################################
    // STEP 1: precision-decompose and send A & B
    // ###########################################

    // No need to use events here, communications from/to the device are serialized

    // decompose A and B on the host (don't start copying yet to get accurate perf numbers)
    de_f64f32(A.ptr, A_h, A_l, A.rows, A.cols, A.ld);
    de_f64f32(B.ptr, B_h, B_l, B.rows, B.cols, B.ld);
    
    auto start = std::chrono::high_resolution_clock::now();

    hello<<<1, 1, 0, s0>>>();
    hello<<<1, 1, 0, s1>>>();
    hello<<<1, 1, 0, s2>>>();
    hello<<<1, 1, 0, s3>>>();

    // copy A_h to GPU
    HANDLE_ERR(cudaMemcpyAsync(dA_h, A_h, A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice, s0));

    // copy B_h to GPU
    HANDLE_ERR(cudaMemcpyAsync(dB_h, B_h, B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice, s2));

    // copy A_l to GPU
    HANDLE_ERR(cudaMemcpyAsync(dA_l, A_l, A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice, s1));

    // copy B_l to GPU
    HANDLE_ERR(cudaMemcpyAsync(dB_l, B_l, B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice, s3));
    HANDLE_ERR(cudaEventRecord(e1, s3));

    // ###################################################
    // STEP 2: convert dA_h and dB_h to double
    // ###################################################

    // launch the kernel in blocks of 256 threads
    // we removed the LD when we decomposed the initial matrices
    // from now on, we can treat the buffers as matrices where ld = rows
    f32tof64_flat<<<ceilDiv(A.rows * A.cols, 256U), 256, 0, s0>>>(dA_h, dA_dgemm, A.rows * A.cols);

    f32tof64_flat<<<ceilDiv(B.rows * B.cols, 256U), 256, 0, s2>>>(dB_h, dB_dgemm, B.rows * B.cols);

    // notify that the conversion is done for dB_h
    HANDLE_ERR(cudaEventRecord(e0, s2));


    // ###################################################
    // STEP 3: dgemm on dA and dB
    // ###################################################

    // wait for s2 to finish converting dB_h into dB_dgemm
    HANDLE_ERR(cudaStreamWaitEvent(s0, e0, 0));

    // perform the dgemm (dRes_dgemm = dA_dgemm * dB_dgemm) on S0
    HANDLE_ERR(cublasSetStream(handle, s0));

    double beta_main = 0.0;

    HANDLE_ERR(cublasDgemm(
        handle,
        convertToCublas(transA), convertToCublas(transB),
        m, n, k,
        &alpha,
        dA_dgemm, checked_cast<int>(A.rows),
        dB_dgemm, checked_cast<int>(B.rows),
        &beta_main,
        dRes_dgemm, checked_cast<int>(C.rows)
    ));
    
    // notify that the dgemm on s0 is done
    HANDLE_ERR(cudaEventRecord(e0, s0));

    // ###################################################
    // STEP 4: decompose dRes_dgemm into dC_h and dC_l
    // ###################################################

    // perform the decomposition immediatly on s0 because
    // it is the stream which performed the dgemm
    extractf32_mixedhl_flat<<<ceilDiv(C.rows * C.cols, 256U), 256, 0, s0>>>(dRes_dgemm, dC_h, dC_l, C.rows * C.cols);
    
    HANDLE_ERR(cudaEventRecord(e0, s0)); // s0 finished the decomposition

    // ###################################################
    // STEP 5: send back dC_h to the host
    // ###################################################

    HANDLE_ERR(cudaMemcpyAsync(C_h, dC_h, C.rows * C.cols * sizeof(float), cudaMemcpyDeviceToHost, s0));

    // ###################################################
    // STEP 6: accumulate sgemm rounds
    // ###################################################

    float sgemm_alpha = 1.0f;
    float sgemm_beta = 1.0f;

    // wait for s2 to finish the dC_l decomposition
    // and perform the sgemm dC_l =  dA_h * dB_l + dC_l
    HANDLE_ERR(cudaStreamWaitEvent(s1, e0, 0));
    HANDLE_ERR(cublasSetStream(handle, s1));
    HANDLE_ERR(cublasSgemm(
        handle,
        convertToCublas(transA), convertToCublas(transB),
        m, n, k,
        &sgemm_alpha,
        dA_l, checked_cast<int>(A.rows),
        dB_h, checked_cast<int>(B.rows),
        &sgemm_beta,
        dC_l, checked_cast<int>(C.rows)
    ));

    // wait for s4 to finish the upload of dC_h
    // and perform the sgemm dC_l = dA_l * dB_h + dC_l
    HANDLE_ERR(cudaStreamWaitEvent(s1, e1, 0));
    HANDLE_ERR(cublasSgemm(
        handle,
        convertToCublas(transA), convertToCublas(transB),
        m, n, k,
        &sgemm_alpha,
        dA_h, checked_cast<int>(A.rows),
        dB_l, checked_cast<int>(B.rows),
        &sgemm_beta,
        dC_l, checked_cast<int>(C.rows)
    ));

    // ###################################################
    // STEP 7: send the result back to the host
    // ###################################################
    
    HANDLE_ERR(cudaMemcpyAsync(C_l, dC_l, C.rows * C.cols * sizeof(float), cudaMemcpyDeviceToHost, s1));

    // ###################################################
    // STEP 8 wait for everything to finish
    // ###################################################

    HANDLE_ERR(cudaStreamSynchronize(s0));
    HANDLE_ERR(cudaStreamSynchronize(s1));
    HANDLE_ERR(cudaStreamSynchronize(s3));
    HANDLE_ERR(cudaStreamSynchronize(s2));

    perf.compute = std::chrono::high_resolution_clock::now() - start;

    // ###################################################
    // STEP 9: cleanup
    // ###################################################

    HANDLE_ERR(cudaStreamDestroy(s1));
    HANDLE_ERR(cudaStreamDestroy(s2));
    HANDLE_ERR(cudaStreamDestroy(s3));

    HANDLE_ERR(cudaEventDestroy(e0));
    HANDLE_ERR(cudaEventDestroy(e1));

    // ###################################################
    // STEP 10: precision-recompose C
    // ###################################################

    re_f64f32(C_h, C_l, C.ptr, C.rows, C.cols, C.ld);

    HANDLE_ERR(cudaFree(dA_h));
    HANDLE_ERR(cudaFree(dA_l));
    HANDLE_ERR(cudaFree(dA_dgemm));
    HANDLE_ERR(cudaFree(dB_h));
    HANDLE_ERR(cudaFree(dB_l));
    HANDLE_ERR(cudaFree(dB_dgemm));
    HANDLE_ERR(cudaFree(dC_h));
    HANDLE_ERR(cudaFree(dC_l));
    HANDLE_ERR(cudaFree(dRes_dgemm));

    HANDLE_ERR(cudaFreeHost(A_h));
    HANDLE_ERR(cudaFreeHost(A_l));
    HANDLE_ERR(cudaFreeHost(B_h));
    HANDLE_ERR(cudaFreeHost(C_h));
    HANDLE_ERR(cudaFreeHost(C_l));

    HANDLE_ERR(cudaDeviceSynchronize());

    return perf;
}

template <typename DataType>
PerfRecord PPMatrix<DataType>::gemm(
    cublasHandle_t handle,
        char transA,
        char transB,
        DataType alpha,
        const PPMatrix<DataType>& A,
        const PPMatrix<DataType>& B,
        DataType beta,
        PPMatrix<DataType>& C
) {
    assert(A.rows == C.rows);
    assert(B.cols == C.cols);
    assert(A.cols == B.rows);

    DataType *dA, *dB, *dC;

    bool use_beta = !is_literal_zero(beta);

    int m = checked_cast<int>((transA == 'N') ? A.rows : A.cols);
    int n = checked_cast<int>((transB == 'N') ? B.cols : B.rows);
    int k = checked_cast<int>((transA == 'N') ? A.cols : A.rows);

    HANDLE_ERR(cudaMalloc(&dA, A.rows * A.cols * sizeof(DataType)));
    HANDLE_ERR(cudaMalloc(&dB, B.rows * B.cols * sizeof(DataType)));
    HANDLE_ERR(cudaMalloc(&dC, C.rows * C.cols * sizeof(DataType)));

    HANDLE_ERR(cudaHostRegister(A.ptr, A.rows * A.cols * sizeof(DataType), cudaHostRegisterDefault));
    HANDLE_ERR(cudaHostRegister(B.ptr, B.rows * B.cols * sizeof(DataType), cudaHostRegisterDefault));

    HANDLE_ERR(cudaHostRegister(C.ptr, C.rows * C.cols * sizeof(DataType), cudaHostRegisterDefault));

    auto start = std::chrono::high_resolution_clock::now();

    HANDLE_ERR(cudaMemcpy(dA, A.ptr, A.rows * A.cols * sizeof(DataType), cudaMemcpyHostToDevice));
    HANDLE_ERR(cudaMemcpy(dB, B.ptr, B.rows * B.cols * sizeof(DataType), cudaMemcpyHostToDevice));

    if (use_beta) {
        HANDLE_ERR(cudaMemcpy(dC, C.ptr, C.rows * C.cols * sizeof(DataType), cudaMemcpyHostToDevice));
    }

    cudaDeviceSynchronize();

    auto h2dDone = std::chrono::high_resolution_clock::now();

    if constexpr (std::is_same_v<DataType, f32>) {
        HANDLE_ERR(cublasSgemm(
            handle,
            convertToCublas(transA), convertToCublas(transB),
            m, n, k,
            &alpha,
            dA, checked_cast<int>(A.rows),
            dB, checked_cast<int>(B.rows),
            &beta,
            dC, checked_cast<int>(C.rows)
        ));
    } else {
        static_assert(std::is_same_v<DataType, f64>, "Unsupported data type (only f32 and f64 are supported).");
        HANDLE_ERR(cublasDgemm(
            handle,
            convertToCublas(transA), convertToCublas(transB),
            m, n, k,
            &alpha,
            dA, checked_cast<int>(A.rows),
            dB, checked_cast<int>(B.rows),
            &beta,
            dC, checked_cast<int>(C.rows)
        ));
    }

    cudaDeviceSynchronize();

    auto computeDone = std::chrono::high_resolution_clock::now();

    HANDLE_ERR(cudaMemcpy(C.ptr, dC, C.rows * C.cols * sizeof(DataType), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();

    auto d2hDone = std::chrono::high_resolution_clock::now();

    HANDLE_ERR(cudaFree(dA));
    HANDLE_ERR(cudaFree(dB));
    HANDLE_ERR(cudaFree(dC));

    HANDLE_ERR(cudaHostUnregister(A.ptr));
    HANDLE_ERR(cudaHostUnregister(B.ptr));

    if (use_beta) {
        HANDLE_ERR(cudaHostUnregister(C.ptr));
    }

    return PerfRecord{ h2dDone - start, computeDone - h2dDone, d2hDone - computeDone };
}

template <typename DataType>
void PPMatrix<DataType>::blasGemm(
    char transA,
    char transB,
    DataType alpha,
    const PPMatrix<DataType>& A,
    const PPMatrix<DataType>& B,
    DataType beta,
    PPMatrix<DataType>& C
) {
    assert(A.rows == C.rows);
    assert(B.cols == C.cols);
    assert(A.cols == B.rows);

    int m = checked_cast<int>((transA == 'N') ? A.rows : A.cols);
    int n = checked_cast<int>((transB == 'N') ? B.cols : B.rows);
    int k = checked_cast<int>((transA == 'N') ? A.cols : A.rows);

    if constexpr (std::is_same_v<DataType, f32>) {
        sgemm_(&transA, &transB, &m, &n, &k, &alpha, A.ptr, &m, B.ptr, &k, &beta, C.ptr, &m);
    } else {
        static_assert(std::is_same_v<DataType, f64>, "Unsupported data type (only f32 and f64 are supported).");
        dgemm_(&transA, &transB, &m, &n, &k, &alpha, A.ptr, &m, B.ptr, &k, &beta, C.ptr, &m);
    }
}

template <typename DataType>
void PPMatrix<DataType>::sub(
    const PPMatrix<DataType>& A,
    const PPMatrix<DataType>& B,
    PPMatrix<DataType>& C
) {
    assert((A.rows == B.rows) && (A.rows == B.rows));
    assert((A.cols == B.cols) && (A.cols == B.cols));

    for (u32 i = 0; i < A.rows; ++i) {
        for (u32 j = 0; j < A.cols; ++j) {
            C.ptr[i * C.ld + j] = A.ptr[i * A.ld + j] - B.ptr[i * B.ld + j];
        }
    }
}

template <typename DataType>
DataType PPMatrix<DataType>::norm(char norm) const {
    int M   = checked_cast<int>(this->rows);
    int N   = checked_cast<int>(this->cols);
    int LD  = checked_cast<int>(this->ld);

    if constexpr (std::is_same_v<DataType, f32>) {
        return slange_(&norm, &M, &N, this->ptr, &LD, nullptr);
    } else {
        static_assert(std::is_same_v<DataType, f64>, "Unsupported data type (only f32 and f64 are supported).");
        return dlange_(&norm, &M, &N, this->ptr, &LD, nullptr);
    }
}


template <typename DataType>
DataType PPMatrix<DataType>::norm2() const {
    char JOBU = 'N';
    char JOBVT = 'N';
    int M   = checked_cast<int>(this->rows);
    int N   = checked_cast<int>(this->cols);
    int LDA = checked_cast<int>(this->ld);

    auto lw = std::min(this->rows, this->cols);
    auto lwork_base = std::min(this->rows, this->cols) * 50;

    auto S = std::vector<DataType>(lw);
    int LDU = 1;
    int LDVT = 1;
    int LWORK = checked_cast<int>(lwork_base);
    auto WORK = std::vector<DataType>(lwork_base);
    int INFO = 0;

    if constexpr (std::is_same_v<DataType, f32>) {
        sgesvd_(&JOBU, &JOBVT, &M, &N, this->ptr, &LDA, S.data(), nullptr, &LDU, nullptr, &LDVT, WORK.data(), &LWORK, &INFO);
    } else {
        static_assert(std::is_same_v<DataType, f64>, "Unsupported data type (only f32 and f64 are supported).");
        dgesvd_(&JOBU, &JOBVT, &M, &N, this->ptr, &LDA, S.data(), nullptr, &LDU, nullptr, &LDVT, WORK.data(), &LWORK, &INFO);
    }

    return S[0];
}

template class PPMatrix<f32>;
template class PPMatrix<f64>;
