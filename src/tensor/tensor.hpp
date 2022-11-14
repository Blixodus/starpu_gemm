#ifndef TENSOR_HPP
#define TENSOR_HPP

static struct starpu_perfmodel tensor_add_perf_model_float = {
  .type = STARPU_REGRESSION_BASED,
  .symbol = "tensor_add_perf_model_float"
};

static struct starpu_perfmodel tensor_add_perf_model_double = {
  .type = STARPU_REGRESSION_BASED,
  .symbol = "tensor_add_perf_model_double"
};

template <typename DataType>
starpu_codelet tensor_add_cl = {
  .cpu_funcs = { tensor_add_cpu_func<DataType> },
#ifdef STARPU_USE_CUDA
  //.cuda_funcs = { tensor_add_cuda_func<DataType> },
  //.cuda_flags = { STARPU_CUDA_ASYNC },
#endif
  .nbuffers = 3,
  .modes = { STARPU_R, STARPU_R, STARPU_RW },
  .model = (std::is_same_v<DataType, float>) ? &tensor_add_perf_model_float : &tensor_add_perf_model_double,
};

template <typename DataType>
starpu_codelet tensor_fill_cl = {
  //.cpu_funcs = { tensor_fill_cpu_func<DataType> },
#ifdef STARPU_USE_CUDA
  .cuda_funcs = { tensor_fill_cuda_func<DataType> },
  //.cuda_flags = { STARPU_CUDA_ASYNC },
#endif
  .nbuffers = 1,
  .modes = { STARPU_W },
};

template <typename DataType>
starpu_codelet tensor_print_cl = {
  //.cpu_funcs = { tensor_print_cpu_func<DataType> },
#ifdef STARPU_USE_CUDA
  .cuda_funcs = { tensor_print_cuda_func<DataType> },
  //.cuda_flags = { STARPU_CUDA_ASYNC },
#endif
  .nbuffers = 1,
  .modes = { STARPU_R },
};

template <typename DataType>
struct Tensor {
  size_t ndim;
  std::vector<unsigned int> dim_size;
  std::vector<unsigned int> ld;
  std::vector<unsigned int> blocks;
  starpu_data_handle_t data_handle;
  DataType * data;

  Tensor(size_t ndim_, std::vector<unsigned int>& dim_size_) : ndim(ndim_), dim_size(dim_size_), ld(ndim, 1), blocks(ndim_, 1) {
    for(int i = 1; i < ndim; i++) { ld[i] = ld[i-1] * dim_size[i-1]; }
    //starpu_ndim_data_register(&data_handle, -1, 0, &ld[0], &dim_size[0], ndim, sizeof(DataType));
    starpu_malloc((void**)&data, (ld[ndim-1]*dim_size[ndim-1])*sizeof(DataType));
    for(int i = 0; i < ld[ndim-1]*dim_size[ndim-1]; i++) { data[i] = -1; }
    starpu_ndim_data_register(&data_handle, STARPU_MAIN_RAM, (uintptr_t)data, &ld[0], &dim_size[0], ndim, sizeof(DataType));
    std::cout << "Created Tensor " << this << " (data_handle=" << data_handle << ")" << std::endl;
  }

  Tensor(size_t ndim_, std::vector<unsigned int>& dim_size_, std::vector<unsigned int>& tile_size) : Tensor(ndim_, dim_size_) {
    partition(tile_size);
  }

  ~Tensor() {
    std::cout << "Destroying Tensor " << this << " (data_handle=" << data_handle << ")" << std::endl;
    if(isPartitioned()) { unpartition(); }
    starpu_data_unregister(data_handle);
    DataType ref_value = data[0];
    std::cout << ref_value << std::endl;
    bool cond = false;
    for(int i = 0; i < ld[ndim-1]*dim_size[ndim-1]; i++) {
      int * truc = (int*)&data[i];
      if((cond && data[i] != ref_value) || (!cond && data[i] == ref_value)) { std::cout << data[i] << " " << i << " " << ld[ndim-1]*dim_size[ndim-1] << std::endl; cond = !cond; }
    }
    std::cout << "Destroyed Tensor" << std::endl;
  }

  void partition(std::vector<unsigned int>& tile_size) {
    if(isPartitioned()) { unpartition(); }
    for(int i = 0; i < ndim; i++) {
      blocks[i] = (dim_size[i] + tile_size[i] - 1)/tile_size[i];
    }
    partitionBlocks();
  }

  void partitionBlocks() {
    std::vector<starpu_data_filter> filters(ndim);
    std::vector<starpu_data_filter *> filters_p(ndim);
    /*
    for(unsigned int i = 0; i < ndim; i++) {
      filters[i] = {
        .filter_func = starpu_ndim_filter_block,
        .nchildren = blocks[i],
        .filter_arg = i
      };
      filters_p[i] = &filters[i];
    }
    */
    //fstarpu_data_map_filters(data_handle, ndim, &filters_p[0]);
  }

  void unpartition() {
    starpu_data_unpartition(data_handle, STARPU_MAIN_RAM);
    blocks = std::vector<unsigned int>(blocks.size(), 1);
  }

  bool isPartitioned() {
    std::vector<unsigned int> no_blocks(blocks.size(), 1);
    return blocks != no_blocks;
  }

  void fill(DataType e) {
    unsigned int nb_blocks = 1;
    for(int i = 0; i < ndim; i++) { nb_blocks *= blocks[i]; }
    {
      std::vector<int> curr_block(ndim, 0);
      for(int i = 0; i < nb_blocks; i++) {
        // Create task for current block
        starpu_data_handle_t block_handle;// = fstarpu_data_get_sub_data(data_handle, ndim, &curr_block[0]);
        int err = starpu_task_insert(&tensor_fill_cl<DataType>,
                                     STARPU_VALUE, &e, sizeof(e),
                                     STARPU_W, block_handle,
                                     0);
        if(err) { throw std::exception(); }
        // Move to next block
        for(int dim = 0; dim < ndim; dim++) {
          curr_block[dim] = (curr_block[dim] < blocks[dim]-1) ? curr_block[dim]+1 : 0;
          if(curr_block[dim]) { break; }
        }
      }
    }
    starpu_task_wait_for_all();
    /*
    {
      std::vector<int> curr_block(ndim, 0);
      for(int i = 0; i < nb_blocks; i++) {
        int block_idx = curr_block[2] * 128 + curr_block[3] * 256 + curr_block[4] * 512;
        // Create task for current block
        starpu_data_handle_t block_handle = fstarpu_data_get_sub_data(data_handle, ndim, &curr_block[0]);
        int err = starpu_task_insert(&tensor_print_cl<DataType>,
                                     STARPU_VALUE, &block_idx, sizeof(block_idx),
                                     STARPU_R, block_handle,
                                     0);
        if(err) { throw std::exception(); }
        // Move to next block
        for(int dim = 0; dim < ndim; dim++) {
          curr_block[dim] = (curr_block[dim] < blocks[dim]-1) ? curr_block[dim]+1 : 0;
          if(curr_block[dim]) { break; }
        }
      }
    }
    */
  }

  Tensor<DataType> operator+(Tensor<DataType>& other) {
    Tensor<DataType> result(ndim, dim_size);
    result.blocks = blocks;
    result.partitionBlocks();
    add(*this, other, result);
    return result;
  }

  /**
   * Do operation C = A + B with tasks
   **/
  void add(Tensor<DataType>& A, Tensor<DataType>& B, Tensor<DataType>& C) {
    assert(A.ndim == B.ndim && B.ndim == C.ndim);
    assert(A.dim_size == B.dim_size && B.dim_size == C.dim_size);
    assert(A.blocks == B.blocks && B.blocks == C.blocks);
    auto & ndim = A.ndim;
    auto & blocks = A.blocks;
    unsigned int nb_blocks = 1;
    for(int i = 0; i < ndim; i++) { nb_blocks *= blocks[i]; }
    
    std::vector<int> curr_block(ndim, 0);
    for(int i = 0; i < nb_blocks; i++) {
      // Create task for current block
      starpu_data_handle_t block_handle_A;// = fstarpu_data_get_sub_data(A.data_handle, ndim, &curr_block[0]);
      starpu_data_handle_t block_handle_B;// = fstarpu_data_get_sub_data(B.data_handle, ndim, &curr_block[0]);
      starpu_data_handle_t block_handle_C;// = fstarpu_data_get_sub_data(C.data_handle, ndim, &curr_block[0]);
      int err = starpu_task_insert(&tensor_add_cl<DataType>,
                                   STARPU_R, block_handle_A,
                                   STARPU_R, block_handle_B,
                                   STARPU_RW, block_handle_C,
                                   0);
      if(err) { throw std::exception(); }
      // Move to next block
      for(int dim = 0; dim < ndim; dim++) {
        curr_block[dim] = (curr_block[dim] < blocks[dim]-1) ? curr_block[dim]+1 : 0;
        if(curr_block[dim]) { break; }
      }
    }
  }
};

#endif
