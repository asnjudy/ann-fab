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

#define NO_GPU throw std::runtime_error("Cannot use GPU in CPU-only Caffe: check mode.\n")

#ifndef CPU_ONLY

#if __CUDA_ARCH__ >= 200
    const int ANNFAB_CUDA_NUM_THREADS = 1024;
#else
    const int ANNFAB_CUDA_NUM_THREADS = 512;
#endif

#endif

}  // namespace annfab


#endif  // ANNFAB_COMMON_HPP_
