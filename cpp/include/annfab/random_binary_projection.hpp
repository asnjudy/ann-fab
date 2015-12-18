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
  explicit RandomBinaryProjection(int projection_count, int batch_size, int dim, int rand_seed);
  virtual ~RandomBinaryProjection();
  
  /** Reset the dimensionality of allowed inputs. Overwrites old random data in the projection matrix */
  void set_dim(int dim);
  
  /** Reset the number of projections. Overwrites old random data in the projection matrix */
  void set_projection_count(int projection_count);
  
  /** Reset the batch size. Data in projection matrix stays the same */
  void set_batch_size(int batch_size);
  
  const vector<Dtype> get_projection_matrix() {
    return vector<Dtype>(_projection_matrix, _projection_matrix + _projection_count * _dim);
  }
  void hash_matrix(Dtype* data);
  int get_dim() const {return _dim;}
  int get_batch_size() const {return _batch_size;}
  int get_projection_count() const {return _projection_count;}
  const char* get_output() const { return _output_chars_h;}
private:
  explicit RandomBinaryProjection() {};
  void _alloc_input_data(bool free_first);
  void _alloc_projection_data(bool free_first);
  void _alloc_output_data(bool free_first);
  int _projection_count;
  int _dim;
  int _batch_size;
  Dtype* _projection_matrix;
  Dtype* _input_data_d;
  Dtype* _output_data_d;
  char* _output_chars_d;
  char* _output_chars_h;
  curandGenerator_t _gen;
  cublasHandle_t _handle;

DISABLE_COPY_AND_ASSIGN(RandomBinaryProjection);
};

}  // namespace annfab

#endif  // ANNFAB_RANDOM_BINARY_PROJECTION_H
