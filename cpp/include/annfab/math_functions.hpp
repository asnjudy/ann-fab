#ifndef ANNFAB_MATH_FUNCTIONS_H_
#define ANNFAB_MATH_FUNCTIONS_H_

#include <cblas.h>

#include "annfab/common.hpp"

namespace annfab {

#ifndef CPU_ONLY

template <typename Dtype>
void annfab_gpu_gemm(cublasHandle_t handle, const cublasOperation_t TransA,
    const cublasOperation_t TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

#endif  // CPU_ONLY

template <typename Dtype>
void annfab_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

}  // namespace annfab

#endif // ANNFAB_MATH_FUNCTIONS_H_