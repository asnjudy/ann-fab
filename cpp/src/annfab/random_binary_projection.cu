#include "annfab/math_functions.hpp"
#include "annfab/random_binary_projection.hpp"
#include "assert.h"


namespace annfab {
  
template <typename Dtype>
__global__ void ZeroOneKernel(const int n, Dtype* input_data, char* output_data) {
  CUDA_KERNEL_LOOP(index, n) {
    output_data[index] = input_data[index] > 0 ? '1' : '0';
  }
}

template<typename Dtype>
void RandomBinaryProjection<Dtype>::hash_matrix_gpu(Dtype* data) {
  // copy data to device
  CUDA_CHECK(cudaMemcpy(_input_data_d, data, sizeof(Dtype) * _dim * _batch_size, cudaMemcpyHostToDevice));
  
  // do the projection
  annfab_gpu_gemm(_handle, CUBLAS_OP_N, CUBLAS_OP_N, _batch_size, _projection_count, _dim,
    Dtype(1.0), _input_data_d, _projection_matrix_d, Dtype(0.0),
    _output_data_d);
  int bla = _projection_count * _batch_size;
  
  // set each number to zero or one
  ZeroOneKernel<Dtype><<<ANNFAB_GET_BLOCKS(bla), ANNFAB_CUDA_NUM_THREADS>>>(bla, _output_data_d, _output_chars_d);
  CUDA_POST_KERNEL_CHECK;
  
  // copy data back to host
  CUDA_CHECK(cudaMemcpy(_output_chars_h, _output_chars_d, sizeof(char) * _batch_size * _projection_count, cudaMemcpyDeviceToHost));
}

template void RandomBinaryProjection<float>::hash_matrix_gpu(float* data);
template void RandomBinaryProjection<double>::hash_matrix_gpu(double* data);

}  // namespace annfab


