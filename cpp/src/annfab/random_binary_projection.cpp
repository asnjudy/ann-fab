#include "annfab/math_functions.hpp"
#include "annfab/random_binary_projection.hpp"
#include "assert.h"

namespace annfab {

#ifndef CPU_ONLY
void gpu_rng_gaussian(curandGenerator_t& gen, const int n, float* r) {
  CURAND_CHECK(curandGenerateNormal(gen, r, n, float(0), float(1)));
}

void gpu_rng_gaussian(curandGenerator_t& gen, const int n, double* r) {
  CURAND_CHECK(curandGenerateNormalDouble(gen, r, n, double(0), double(1)));
}
#endif
  
template<typename Dtype>
RandomBinaryProjection<Dtype>::RandomBinaryProjection(int projection_count, int batch_size, int dim, int rand_seed, bool use_gpu)
  : _projection_count(projection_count), _dim(dim), _batch_size(batch_size), _GPU(use_gpu) {
  assert(dim > 0);
  assert(projection_count > 0);
  assert(batch_size > 0);
  if (_GPU) {
#ifdef CPU_ONLY
    NO_GPU;
#else
    CURAND_CHECK(curandCreateGenerator(&_gen, 
                  CURAND_RNG_PSEUDO_DEFAULT));
    // set GPU RNG seed
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(_gen, 
                rand_seed));
    // create handle
    CUBLAS_CHECK(cublasCreate(&_handle));
#endif
  } else {
    // set CPU RNG seed
  }
  _alloc_input_data(false);
  _alloc_projection_data(false);
  _alloc_output_data(false);
}

template<typename Dtype>
RandomBinaryProjection<Dtype>::~RandomBinaryProjection() {
  free(_output_chars_h);
  if (_GPU) {
#ifndef CPU_ONLY
    cudaFree(_projection_matrix_d);
    cudaFree(_output_data_d);
    cudaFree(_output_chars_d);
    cudaFree(_input_data_d);
    CURAND_CHECK(curandDestroyGenerator(_gen));
    cublasDestroy(_handle);
#endif
  } else {
    free(_projection_matrix_h); 
  }
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
void RandomBinaryProjection<Dtype>::hash_matrix_cpu(Dtype* data) {
  
}

template<typename Dtype>
void RandomBinaryProjection<Dtype>::_alloc_projection_data(bool free_first) {
  //for random number generation, we must generate an even number of numbers
  int proj_dim = _dim * _projection_count;
  if (proj_dim % 2 != 0)
    ++proj_dim;
    
  if(_GPU) {
#ifndef CPU_ONLY
    // free the last projection matrix if needed
    if(free_first)
      cudaFree(_projection_matrix_d);
    CUDA_CHECK(cudaMalloc(&_projection_matrix_d, sizeof(Dtype) * proj_dim));
    gpu_rng_gaussian(_gen, proj_dim, _projection_matrix_d);
#endif
  } else {
    if(free_first)
      free(_projection_matrix_h);
    _projection_matrix_h = (Dtype*)malloc(sizeof(Dtype) * proj_dim);
    if (!_projection_matrix_h)
      throw std::runtime_error("RandomBinaryProjection::_alloc_projection_data: allocation failed\n");
  }
}

template<typename Dtype>
void RandomBinaryProjection<Dtype>::_alloc_input_data(bool free_first) {
#ifndef CPU_ONLY
  if(_GPU) {
    if(free_first)
      cudaFree(_input_data_d); 
    
    // create new input data array
    CUDA_CHECK(cudaMalloc(&_input_data_d, sizeof(Dtype) * _dim * _batch_size));
  }
#endif
}

template<typename Dtype>
void RandomBinaryProjection<Dtype>::_alloc_output_data(bool free_first) {
  if(free_first) {
     free(_output_chars_h);
  }
  
  _output_chars_h = (char*)malloc(sizeof(char) * _batch_size * _projection_count);
  if (!_output_chars_h)
    throw std::runtime_error("RandomBinaryProjection::_alloc_output_data: allocation failed\n");
  
#ifndef CPU_ONLY
  if(_GPU) {
    if(free_first) {
      cudaFree(_output_data_d);
      cudaFree(_output_chars_d);
    }
    CUDA_CHECK(cudaMalloc(&_output_data_d, sizeof(Dtype) * _batch_size * _projection_count));
    CUDA_CHECK(cudaMalloc(&_output_chars_d, sizeof(char) * _batch_size * _projection_count));
  }
#endif
}

template RandomBinaryProjection<float>::RandomBinaryProjection(int projection_count, int batch_size, int dim, int rand_seed, bool use_gpu);
template RandomBinaryProjection<double>::RandomBinaryProjection(int projection_count, int batch_size, int dim, int rand_seed, bool use_gpu);
template void RandomBinaryProjection<float>::set_dim(int dim);
template void RandomBinaryProjection<double>::set_dim(int dim);
template void RandomBinaryProjection<float>::set_projection_count(int projection_count);
template void RandomBinaryProjection<double>::set_projection_count(int projection_count);
template void RandomBinaryProjection<float>::set_batch_size(int batch_size);
template void RandomBinaryProjection<double>::set_batch_size(int batch_size);
template void RandomBinaryProjection<float>::hash_matrix_cpu(float* data);
template void RandomBinaryProjection<double>::hash_matrix_cpu(double* data);

}  // namespace annfab
