#include "annfab/math_functions.hpp"
#include "annfab/random_binary_projection.hpp"

#include <future>
#include <thread>

#ifndef CPU_ONLY
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
#include "annfab/annfab_cuda_utils.hpp"
#endif

#include "assert.h"

namespace annfab {

#ifndef CPU_ONLY
void gpu_rng_gaussian(curandGenerator_t& gen, const int n, float* r) {
  assert_on_cuda_error(curandGenerateNormal(gen, r, n, float(0), float(1)));
}

void gpu_rng_gaussian(curandGenerator_t& gen, const int n, double* r) {
  assert_on_cuda_error(curandGenerateNormalDouble(gen, r, n, double(0), double(1)));
}
#endif

template <typename Dtype, typename Generator=std::mt19937>
void cpu_rng_gaussian(const int n, int seed, Dtype* r) {
  const int NUM_THREADS = 4;
  std::vector<std::future<void> > f(NUM_THREADS);
  int loop_length = n / NUM_THREADS;
  int offset = 0;
  for (int i = 0; i < NUM_THREADS; ++i) {
    if (i == NUM_THREADS - 1)
      loop_length = n - offset;
    auto mylambda = [=]() -> void {
      Generator gen(seed);
      std::normal_distribution<Dtype> d(0,1);
      for (int j = offset; j < offset + loop_length; ++j)
        r[j] = d(gen);
    };
    f[i] = std::async(std::launch::async, mylambda);
    offset += loop_length;
    seed += 100;
  }
  for (int i = 0; i < NUM_THREADS; ++i) {
    f[i].wait();
  }
}
  
template<typename Dtype>
RandomBinaryProjection<Dtype>::RandomBinaryProjection(int projection_count, int batch_size, int dim, int rand_seed, bool use_gpu)
  : _projection_count(projection_count), _dim(dim), _batch_size(batch_size), _rand_seed(rand_seed),  _GPU(use_gpu) {
  assert(dim > 0);
  assert(projection_count > 0);
  assert(batch_size > 0);
  if (_GPU) {
#ifdef CPU_ONLY
    NO_GPU;
#else
    assert_on_cuda_error(curandCreateGenerator(&_gpu_gen,
                         CURAND_RNG_PSEUDO_DEFAULT));
    // create handle
    assert_on_cuda_error(cublasCreate(&_handle));
#endif
  }
  _alloc_input_data(false);
  _alloc_projection_data(false);
  _alloc_output_data(false);
}

template<typename Dtype>
RandomBinaryProjection<Dtype>::~RandomBinaryProjection() {
  if (_GPU) {
#ifndef CPU_ONLY
    cudaFree(_projection_matrix_d);
    cudaFree(_output_data_d);
    cudaFree(_output_chars_d);
    cudaFree(_input_data_d);
    assert_on_cuda_error(curandDestroyGenerator(_gpu_gen));
    cublasDestroy(_handle);
#endif
  } else {
    free(_output_data_h);
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
void RandomBinaryProjection<Dtype>::_hash_matrix_cpu(const Dtype* query, char* hash) {
  // do the projection
  annfab_cpu_gemm(CblasNoTrans,CblasNoTrans, _batch_size, _projection_count, _dim,
    Dtype(1.0), query, _projection_matrix_h, Dtype(0.0),
    _output_data_h);
  
  // now convert this to a bunch of characters
  for (int i = 0; i < _projection_count * _batch_size; ++i) {
    hash[i] = _output_data_h[i] > 0 ? '1' : '0';
  }
}

template<typename Dtype>
void RandomBinaryProjection<Dtype>::_alloc_projection_data(bool free_first) {
  //for random number generation, we must generate an even number of numbers
  int proj_dim = _dim * _projection_count;

  if(_GPU) {
#ifndef CPU_ONLY
    // free the last projection matrix if needed
    if(free_first)
      cudaFree(_projection_matrix_d);
    // for whatever reason, the projection dim must be even in the GPU case
    if (proj_dim % 2 != 0)
      ++proj_dim;
    assert_on_cuda_error(cudaMalloc(&_projection_matrix_d, sizeof(Dtype) * proj_dim));
    // (re)set GPU RNG seed
    assert_on_cuda_error(curandSetPseudoRandomGeneratorSeed(_gpu_gen,
                         _rand_seed));
    gpu_rng_gaussian(_gpu_gen, proj_dim, _projection_matrix_d);
#endif
  } else {
    if(free_first)
      free(_projection_matrix_h);
    _projection_matrix_h = (Dtype*)malloc(sizeof(Dtype) * proj_dim);
    if (!_projection_matrix_h)
      throw std::runtime_error("RandomBinaryProjection::_alloc_projection_data: allocation failed\n");
    cpu_rng_gaussian(proj_dim, _rand_seed, _projection_matrix_h);
  }
}

template<typename Dtype>
void RandomBinaryProjection<Dtype>::_alloc_input_data(bool free_first) {
#ifndef CPU_ONLY
  if(_GPU) {
    if(free_first)
      cudaFree(_input_data_d);

    // create new input data array
    assert_on_cuda_error(cudaMalloc(&_input_data_d, sizeof(Dtype) * _dim * _batch_size));
  }
#endif
}

template<typename Dtype>
void RandomBinaryProjection<Dtype>::_alloc_output_data(bool free_first) {
  if(_GPU) {
#ifndef CPU_ONLY
    if(free_first) {
      cudaFree(_output_data_d);
      cudaFree(_output_chars_d);
    }
    assert_on_cuda_error(cudaMalloc(&_output_data_d, sizeof(Dtype) * _batch_size * _projection_count));
    assert_on_cuda_error(cudaMalloc(&_output_chars_d, sizeof(char) * _batch_size * _projection_count));
#endif
  } else {
    if(free_first) {
      free(_output_data_h);
    }
    _output_data_h = (Dtype*)malloc(sizeof(Dtype) * _batch_size * _projection_count);
    if (!_output_data_h)
      throw std::runtime_error("RandomBinaryProjection::_alloc_output_data: _output_data_h allocation failed\n");
  }
}

template RandomBinaryProjection<float>::RandomBinaryProjection(int projection_count, int batch_size, int dim, int rand_seed, bool use_gpu);
template RandomBinaryProjection<double>::RandomBinaryProjection(int projection_count, int batch_size, int dim, int rand_seed, bool use_gpu);
template void RandomBinaryProjection<float>::set_dim(int dim);
template void RandomBinaryProjection<double>::set_dim(int dim);
template void RandomBinaryProjection<float>::set_projection_count(int projection_count);
template void RandomBinaryProjection<double>::set_projection_count(int projection_count);
template void RandomBinaryProjection<float>::set_batch_size(int batch_size);
template void RandomBinaryProjection<double>::set_batch_size(int batch_size);
template void RandomBinaryProjection<float>::_hash_matrix_cpu(const float* query, char* hash);
template void RandomBinaryProjection<double>::_hash_matrix_cpu(const double* query, char* hash);

}  // namespace annfab
