#ifndef ANNFAB_RANDOM_BINARY_PROJECTION_H
#define ANNFAB_RANDOM_BINARY_PROJECTION_H

#include <random>

#include "annfab/common.hpp"
namespace annfab {

template<typename Dtype>

/**
 * @brief Perform random binary projections on both CPU and GPU
 * When constructed, the class allocates memory to perform random binary projections
 * and creates a random projection matrix with standard normal values. In this way,
 * query vectors can be hashed without any allocation / setup overhead
 * 
 * The hashing of a query matrix is done by 
 * 
 * Q * P = X
 * 
 * Wher Q is a query matrix, whith each row a query vector, P is the random projection
 * matrix and X is a matrix of characters, 0 and 1 with each row the hash of a query vector
 * TODO: think about letting the user specify pass in an array where the results should be written
 */
class RandomBinaryProjection {
public:
  /**
   * @brief Constructor of the RandomBinaryProjection class
   * @param projection_count Number of projections to calculate, i.e. length of resulting key
   * @param batch_size Number of query vectors to process in parallel
   * @param dim Length of a query vector
   * @param rand_seed Initialize rng with this seed
   * @param use_gpu hash with the CPU or GPU implementation
   */
  explicit RandomBinaryProjection(int projection_count, int batch_size, int dim, int rand_seed, bool use_gpu);
  virtual ~RandomBinaryProjection();
  
  /** @brief Reset the dimensionality of allowed inputs. Overwrites old random data in the projection matrix */
  void set_dim(int dim);
  
  /** @brief Reset the number of projections. Overwrites old random data in the projection matrix */
  void set_projection_count(int projection_count);
  
  /** @brief Reset the batch size. Data in projection matrix stays the same */
  void set_batch_size(int batch_size);
  
  /** 
   * @brief hash the matrix contained in data, either with CPU or GPU and return the hash matrix
   * @param[in] query row dominated query matrix where each row is one query vector
   * @param[out] hash row dominated hash matrix where each row is the hash of one query vector.
   *                  The array should be projection_count * batch_size long
   */
  void hash_matrix(const Dtype* query, char* hash);
  int get_dim() const {return _dim;}
  int get_batch_size() const {return _batch_size;}
  int get_projection_count() const {return _projection_count;}
private:
  // disable default constructor
  explicit RandomBinaryProjection() {};
  void _hash_matrix_cpu(const Dtype* query, char* hash);
  void _alloc_input_data(bool free_first);       /// allocate the space for input data
  void _alloc_projection_data(bool free_first);  /// allocate the space for the projection matrix and the like
  void _alloc_output_data(bool free_first);      /// allocate the space for the results of the projection
  int _projection_count;                         /// The size of the resulting hash vector
  int _dim;                                      /// The dimensionality of the vectors to be projected
  int _batch_size;                               /// Number of vectors that will be projected in parallel
  int _rand_seed;                                /// The seed used to initialize the random number generator
  Dtype* _output_data_h;                         /// Array to save result of the matrix multiplication
  bool _GPU;                                     /// Flag telling us to use the CPU or GPU implementation
  Dtype* _projection_matrix_h;                   /// pointer to the projection matrix if it is on the host
  shared_ptr<std::mt19937> _cpu_gen;             /// pointer to CPU based random number generator
  
#ifndef CPU_ONLY
  void _hash_matrix_gpu(const Dtype* query, char* hash);
  Dtype* _projection_matrix_d;                   /// pointer to the projection matrix if it is on the device
  Dtype* _input_data_d;                          /// space on device for the query matrix to be copied to
  Dtype* _output_data_d;                         /// space on device for the matrix multiplication to be saved to
  char* _output_chars_d;                         /// space on device for the hash to be saved to before copy
  curandGenerator_t _gpu_gen;                    /// generator on device
  cublasHandle_t _handle;                        /// handle to cublas
#endif

DISABLE_COPY_AND_ASSIGN(RandomBinaryProjection);
};

template <typename Dtype>
void RandomBinaryProjection<Dtype>::hash_matrix(const Dtype* query, char* hash) {
  if (_GPU) {
#ifndef CPU_ONLY
    _hash_matrix_gpu(query, hash);
#endif
  } else {
    _hash_matrix_cpu(query, hash);
  }
}

}  // namespace annfab

#endif  // ANNFAB_RANDOM_BINARY_PROJECTION_H
