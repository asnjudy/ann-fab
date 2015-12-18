#ifndef ANNFAB_MATH_FUNCTIONS_H_
#define ANNFAB_MATH_FUNCTIONS_H_

#include "annfab/common.hpp"

namespace annfab {

template <typename Dtype>
void annfab_gpu_gemm(cublasHandle_t handle, const cublasOperation_t TransA,
    const cublasOperation_t TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

}  // namespace annfab


#endif // ANNFAB_MATH_FUNCTIONS_H_