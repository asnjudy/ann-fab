#ifndef ANNFAB_RANDOM_BINARY_PROJECTION_H
#define ANNFAB_RANDOM_BINARY_PROJECTION_H

#include "annfab/common.hpp"

namespace annfab {

template<typename Dtype>
class RandomBinaryProjection {
public:
  /**
   * Constructor of the RandomBinaryProjection class
   * @param projection_count Number of projections to calculate, i.e. length of resulting key
   * @param batch_size Number of query vectors to process in parallel
   * @param dim Length of a query vector
   * @param rand_seed Initialize rng with this seed
   */
  explicit RandomBinaryProjection(int projection_count, int batch_size, int dim, int rand_seed, bool use_gpu);
  virtual ~RandomBinaryProjection();
  
  /** Reset the dimensionality of allowed inputs. Overwrites old random data in the projection matrix */
  void set_dim(int dim);
  
  /** Reset the number of projections. Overwrites old random data in the projection matrix */
  void set_projection_count(int projection_count);
  
  /** Reset the batch size. Data in projection matrix stays the same */
  void set_batch_size(int batch_size);
  
  /** @brief hash the matrix contained in data, either with CPU or GPU */
  void hash_matrix(Dtype* data);
  int get_dim() const {return _dim;}
  int get_batch_size() const {return _batch_size;}
  int get_projection_count() const {return _projection_count;}
  const char* get_output() const { return _output_chars_h;}
private:
  // disable default constructor
  explicit RandomBinaryProjection() {};
  void hash_matrix_cpu(Dtype* data);
  void _alloc_input_data(bool free_first);       /// allocate the space for input data
  void _alloc_projection_data(bool free_first);  /// allocate the space for the projection matrix and the like
  void _alloc_output_data(bool free_first);      /// allocate the space for the results of the projection
  int _projection_count;                         /// The size of the resulting hash vector
  int _dim;                                      /// The dimensionality of the vectors to be projected
  int _batch_size;                               /// Number of vectors that will be projected in parallel
  char* _output_chars_h;
  bool _GPU;                                     /// Flag telling us to use the CPU or GPU implementation
  Dtype* _projection_matrix_h;                   /// pointer to the projection matrix if it is on the host

#ifndef CPU_ONLY
  void hash_matrix_gpu(Dtype* data);
  Dtype* _projection_matrix_d;                   /// pointer to the projection matrix if it is on the device
  Dtype* _input_data_d;
  Dtype* _output_data_d;
  char* _output_chars_d;
  curandGenerator_t _gen;
  cublasHandle_t _handle;
#endif

DISABLE_COPY_AND_ASSIGN(RandomBinaryProjection);
};


template <typename Dtype>
void RandomBinaryProjection<Dtype>::hash_matrix(Dtype* data) {
  if (_GPU) {
#ifndef CPU_ONLY
    hash_matrix_gpu(data);
#endif
  } else {
    hash_matrix_cpu(data);
  }
}

}  // namespace annfab

#endif  // ANNFAB_RANDOM_BINARY_PROJECTION_H
