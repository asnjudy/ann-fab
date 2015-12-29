#include "annfab/common.hpp"
#include "annfab/math_functions.hpp"
#include "annfab/annfab_cuda_utils.hpp"

namespace annfab {

template <>
void annfab_gpu_gemm<float>(cublasHandle_t handle, const cublasOperation_t TransA,
    const cublasOperation_t TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CUBLAS_OP_N) ? K : M;
  int ldb = (TransB == CUBLAS_OP_N) ? N : K;
  assert_on_cuda_error(cublasSgemm(handle, TransB, TransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void annfab_gpu_gemm<double>(cublasHandle_t handle, const cublasOperation_t TransA,
    const cublasOperation_t TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CUBLAS_OP_N) ? K : M;
  int ldb = (TransB == CUBLAS_OP_N) ? N : K;
  assert_on_cuda_error(cublasDgemm(handle, TransB, TransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

}  // name