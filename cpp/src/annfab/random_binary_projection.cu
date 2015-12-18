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

void gpu_rng_gaussian(curandGenerator_t& gen, const int n, float* r) {
  CURAND_CHECK(curandGenerateNormal(gen, r, n, float(0), float(1)));
}

void gpu_rng_gaussian(curandGenerator_t& gen, const int n, double* r) {
  CURAND_CHECK(curandGenerateNormalDouble(gen, r, n, double(0), double(1)));
}

template<typename Dtype>
RandomBinaryProjection<Dtype>::RandomBinaryProjection(int projection_count, int batch_size, int dim, int rand_seed)
  : _projection_count(projection_count), _dim(dim), _batch_size(batch_size) {
  assert(dim > 0);
  assert(projection_count > 0);
  assert(batch_size > 0);
  
  CURAND_CHECK(curandCreateGenerator(&_gen, 
                CURAND_RNG_PSEUDO_DEFAULT));
  /* Set seed */
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(_gen, 
               rand_seed));
  // create handle
  CUBLAS_CHECK(cublasCreate(&_handle));
  
  _alloc_input_data(false);
  _alloc_projection_data(false);
  _alloc_output_data(false);
}


template<typename Dtype>
RandomBinaryProjection<Dtype>::~RandomBinaryProjection() {
  cudaFree(_projection_matrix);
  cudaFree(_output_data_d);
  cudaFree(_output_chars_d);
  free(_output_chars_h);
  cudaFree(_input_data_d);
  CURAND_CHECK(curandDestroyGenerator(_gen));
  cublasDestroy(_handle);
}

template<typename Dtype>
void RandomBinaryProjection<Dtype>::set_dim(int dim) {
  assert(dim > 0);
  if (_dim != dim) {
    _dim = dim;
    _alloc_projection_data(true);
    _alloc_input_data(true);
  }
}

template<typename Dtype>
void RandomBinaryProjection<Dtype>::set_projection_count(int projection_count) {
  assert(projection_count > 0);
  if (_projection_count != projection_count) {
    _projection_count = projection_count;
    _alloc_projection_data(true);
    _alloc_output_data(true);
  }
}

template<typename Dtype>
void RandomBinaryProjection<Dtype>::set_batch_size(int batch_size) {
  assert(batch_size > 0);
  if (_batch_size != batch_size) {
    _batch_size = batch_size;
    _alloc_input_data(true);
    _alloc_output_data(true);
  }
}

template<typename Dtype>
void RandomBinaryProjection<Dtype>::hash_matrix(Dtype* data) {
  // copy data to device
  CUDA_CHECK(cudaMemcpy(_input_data_d, data, sizeof(Dtype) * _dim * _batch_size, cudaMemcpyHostToDevice));
  
  // do the projection
  annfab_gpu_gemm(_handle, CUBLAS_OP_N, CUBLAS_OP_N, _batch_size, _projection_count, _dim,
    Dtype(1.0), _input_data_d, _projection_matrix, Dtype(0.0),
    _output_data_d);
  int bla = _projection_count * _batch_size;
  
  // set each number to zero or one
  ZeroOneKernel<Dtype><<<ANNFAB_GET_BLOCKS(bla), ANNFAB_CUDA_NUM_THREADS>>>(bla, _output_data_d, _output_chars_d);
  CUDA_POST_KERNEL_CHECK;
  
  // copy data back to host
  CUDA_CHECK(cudaMemcpy(_output_chars_h, _output_chars_d, sizeof(char) * _batch_size * _projection_count, cudaMemcpyDeviceToHost));
}

template<typename Dtype>
void RandomBinaryProjection<Dtype>::_alloc_projection_data(bool free_first) {
  // free the last projection matrix if needed
  if(free_first)
    cudaFree(_projection_matrix);
  //for random number generation, we must generate an even number of numbers
  int proj_dim = _dim * _projection_count;
  if (proj_dim % 2 != 0)
    ++proj_dim;
  CUDA_CHECK(cudaMalloc(&_projection_matrix, sizeof(Dtype) * proj_dim));
  gpu_rng_gaussian(_gen, proj_dim, _projection_matrix);
}

template<typename Dtype>
void RandomBinaryProjection<Dtype>::_alloc_input_data(bool free_first) {
  if(free_first)
    cudaFree(_input_data_d); 
    
  // create new input data array
  CUDA_CHECK(cudaMalloc(&_input_data_d, sizeof(Dtype) * _dim * _batch_size));
}

template<typename Dtype>
void RandomBinaryProjection<Dtype>::_alloc_output_data(bool free_first) {
  if(free_first) {
    cudaFree(_output_data_d);
    cudaFree(_output_chars_d);
    free(_output_chars_h);
    
  }
  _output_chars_h = (char*)malloc(sizeof(char) * _batch_size * _projection_count);
  if (!_output_chars_h)
    throw std::runtime_error("RandomBinaryProjection::_alloc_output_data: allocation failed\n");
  CUDA_CHECK(cudaMalloc(&_output_data_d, sizeof(Dtype) * _batch_size * _projection_count));
  CUDA_CHECK(cudaMalloc(&_output_chars_d, sizeof(char) * _batch_size * _projection_count));
}

template RandomBinaryProjection<float>::RandomBinaryProjection(int projection_count, int batch_size, int dim, int rand_seed);
template RandomBinaryProjection<double>::RandomBinaryProjection(int projection_count, int batch_size, int dim, int rand_seed);
template void RandomBinaryProjection<float>::set_dim(int dim);
template void RandomBinaryProjection<double>::set_dim(int dim);
template void RandomBinaryProjection<float>::set_projection_count(int projection_count);
template void RandomBinaryProjection<double>::set_projection_count(int projection_count);
template void RandomBinaryProjection<float>::set_batch_size(int batch_size);
template void RandomBinaryProjection<double>::set_batch_size(int batch_size);
template void RandomBinaryProjection<float>::hash_matrix(float* data);
template void RandomBinaryProjection<double>::hash_matrix(double* data);

}  // namespace annfab


