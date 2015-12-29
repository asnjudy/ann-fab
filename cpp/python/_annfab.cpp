#include <Python.h>  // NOLINT(build/include_alpha)

// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/make_shared.hpp>
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/arrayobject.h>

// these need to be included after boost on OS X
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)
#include <fstream>  // NOLINT

#include "annfab/random_binary_projection.hpp"
#include "annfab/common.hpp"

// Temporary solution for numpy < 1.7 versions: old macro, no promises.
// You're strongly advised to upgrade to >= 1.7.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define PyArray_SetBaseObject(arr, x) (PyArray_BASE(arr) = (x))
#endif

namespace bp = boost::python;

namespace annfab {
  
// For Python, for now, we'll just always use float as the type.
typedef float Dtype;
const int NPY_DTYPE = NPY_FLOAT32;
  
shared_ptr<RandomBinaryProjection<Dtype> > RBP_Init(int projection_count, int batch_size, int dim, int rand_seed, bool use_gpu) {
  shared_ptr<RandomBinaryProjection<Dtype> > rbp(new RandomBinaryProjection<Dtype>(projection_count, batch_size, dim, rand_seed, use_gpu));
  return rbp;
}

bp::object Hash_Matrix(RandomBinaryProjection<Dtype>& rbp, bp::object data_obj) {
  if (rbp.get_dim() == 0)
    throw std::runtime_error("you must set the dimensionality before starting");
  PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(data_obj.ptr());
  if (!(PyArray_FLAGS(arr) && NPY_ARRAY_C_CONTIGUOUS)) 
    throw std::runtime_error("In annfab::Hash_Matrix array must be C contiguous");
  if (PyArray_NDIM(arr) != 2) 
    throw std::runtime_error("In annfab::Hash_Matrix array must be 2-d");
  if (PyArray_TYPE(arr) != NPY_FLOAT32) 
    throw std::runtime_error("In annfab::Hash_Matrix array must be float32");
  if (PyArray_DIMS(arr)[0] != rbp.get_batch_size())
    throw std::runtime_error("In annfab::Hash_Matrix input shape[0] must equal batch size");
  if (PyArray_DIMS(arr)[1] != rbp.get_dim())
    throw std::runtime_error("In annfab::Hash_Matrix input shape[1] must equal dim");
  float* float_arr = static_cast<float*>(PyArray_DATA(arr));
  rbp.hash_matrix(float_arr);
  npy_intp dims[2];
  dims[0] = rbp.get_batch_size();
  dims[1] = rbp.get_projection_count();
  PyObject *o = PyArray_SimpleNew(2, dims, NPY_CHAR);
  memcpy(static_cast<float*>(PyArray_DATA((PyArrayObject*)o)),
         rbp.get_output(), dims[0] * dims[1] *sizeof(char));
  bp::handle<> h(o);
  return bp::object(h);
}
  
BOOST_PYTHON_MODULE(_annfab) {
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
  import_array();
  
  bp::class_<RandomBinaryProjection<Dtype>, shared_ptr<RandomBinaryProjection<Dtype> >, boost::noncopyable>("RandomBinaryProjection", bp::no_init)
    .def("__init__", bp::make_constructor(&RBP_Init))
    .def("set_dim", &RandomBinaryProjection<Dtype>::set_dim)
    .def("set_projection_count", &RandomBinaryProjection<Dtype>::set_projection_count)
    .def("set_batch_size", &RandomBinaryProjection<Dtype>::set_batch_size)
    .def("hash_matrix", &Hash_Matrix)
    .add_property("dim", &RandomBinaryProjection<Dtype>::get_dim)
    .add_property("projection_count", &RandomBinaryProjection<Dtype>::get_projection_count)
    .add_property("batch_size", &RandomBinaryProjection<Dtype>::get_batch_size);
}
}  // namespace annfab