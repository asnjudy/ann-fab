#include "annfab/math_functions.hpp"
#include "annfab/random_binary_projection.hpp"
#include "assert.h"

#ifndef CPU_ONLY
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
#include "annfab/annfab_cuda_utils.hpp"
#endif

namespace annfab {

template <typename Dtype>
__global__ void ZeroOneKernel(const int n, Dtype* input_data, char* output_data) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
  {
    output_data[i] = input_data[i] > 0 ? '1' : '0';
  }
}

template<typename Dtype>
void RandomBinaryProjection<Dtype>::_hash_matrix_gpu(const Dtype* query, char* hash) {
  // copy data to device
  assert_on_cuda_error(cudaMemcpy(_input_data_d, query, sizeof(Dtype) * _dim * _batch_size, cudaMemcpyHostToDevice));

  // Perform the projection using a matrix-matrix multiplication
  annfab_gpu_gemm(_handle, CUBLAS_OP_N, CUBLAS_OP_N, _batch_size, _projection_count, _dim,
    Dtype(1.0), _input_data_d, _projection_matrix_d, Dtype(0.0),
    _output_data_d);
  int N = _projection_count * _batch_size;

  // Create a binary string based on the values of the output vector.
  // TODO: Use a thrust transform here.
  int num_blocks = get_num_blocks(N, ANNFAB_CUDA_NUM_THREADS);
  ZeroOneKernel<Dtype><<<num_blocks, ANNFAB_CUDA_NUM_THREADS>>>(N, _output_data_d, _output_chars_d);
  assert_on_kernel_result();

  // copy data back to host
  assert_on_cuda_error(cudaMemcpy(hash, _output_chars_d, sizeof(char) * N, cudaMemcpyDeviceToHost));
}

// Instantiate float and double version of the functions above.
template void RandomBinaryProjection<float>::_hash_matrix_gpu(const float* query, char* hash);
template void RandomBinaryProjection<double>::_hash_matrix_gpu(const double* query, char* hash);

}  // namespace annfab


