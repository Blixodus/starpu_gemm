#pragma once

#include <vector>
#include <array>
#include <algorithm>

#include "starpu_mpi.h"

#include "../util/helper.hpp"

static constexpr int enable_cpu = 1;
static constexpr int enable_gpu = 1;

static int can_execute(unsigned workerid, struct starpu_task* task, unsigned nimpl) {
    if (starpu_worker_get_type(safe_cast<int>(workerid)) == STARPU_CPU_WORKER) {
        return enable_cpu;
    }

    return enable_gpu;
}

template <typename DT, u32 NDIMS>
class Tensor {
    public:
        Tensor(std::array<u32, NDIMS> dims, u32 block_size) :
            dims(dims),
            handles(safe_cast<typename decltype(handles)::size_type>(std::reduce(dims.begin(), dims.end(), 1, std::multiplies<u32>())))
        {
            int rank, size;
            starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
            starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

            for (u32 dim: dims) {
                
            }
        }

        ~Tensor() {

        }

        starpu_data_handle_t& get(std::array<u32, NDIMS>& idx) {
            return handles[dims_to_idx(idx)];
        }

    private:
        u32 dims_to_idx(std::array<u32, NDIMS>& idx) {
            return std::reduce(idx.begin(), idx.end(), 0, std::multiplies<u32>());
        }

        std::array<u32, NDIMS> dims;
        std::vector<starpu_data_handle_t> handles;
};

template <typename DT, u32 NDIMS>
Tensor<DT, NDIMS> mk_tensor(std::array<u32, NDIMS> dims, u32 block_size) {
    return Tensor<DT, NDIMS>(dims, block_size);
}