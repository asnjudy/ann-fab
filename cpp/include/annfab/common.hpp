#ifndef ANNFAB_COMMON_HPP_
#define ANNFAB_COMMON_HPP_

#include <assert.h>
#include <climits>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#ifndef CPU_ONLY
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
#endif  // CPU_ONLY

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

namespace annfab {

// Common functions and classes from std that annfab often uses.
using std::ios;
using std::isnan;
using std::isinf;
using std::iterator;
using std::make_pair;
using std::map;
using std::pair;
using std::set;
using std::shared_ptr;
using std::string;
using std::stringstream;
using std::vector;
using std::cout;
using std::endl;

#ifndef CPU_ONLY

#if __CUDA_ARCH__ >= 200
    const int ANNFAB_CUDA_NUM_THREADS = 1024;
#else
    const int ANNFAB_CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for threads.
inline int ANNFAB_GET_BLOCKS(const int N) {
  return (N + ANNFAB_CUDA_NUM_THREADS - 1) / ANNFAB_CUDA_NUM_THREADS;
}

#endif

}  // namespace annfab



#ifndef CPU_ONLY
//
// CUDA macros
//

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error !=  cudaSuccess) {std::cout << " " << cudaGetErrorString(error) << std::endl; assert(false); } \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    if (status !=  CUBLAS_STATUS_SUCCESS) {std::cout << " " << annfab::cublasGetErrorString(status) << std::endl; assert(false); } \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    if (status !=  CURAND_STATUS_SUCCESS) { std::cout << annfab::curandGetErrorString(status) << std::endl; assert(false); } \
  } while (0)

#endif  // CPU_ONLY

namespace annfab {

const char* cublasGetErrorString(cublasStatus_t error);
const char* curandGetErrorString(curandStatus_t error);

}  // namespace annfab

#endif  // ANNFAB_COMMON_HPP_
