#include "trofimov_n_linear_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>

#include "trofimov_n_linear_topology/common/include/common.hpp"

namespace trofimov_n_linear_topology {

TrofimovNLinearTopologyMPI::TrofimovNLinearTopologyMPI(const InType &in)
    : linear_comm_(MPI_COMM_NULL), rank_(0), size_(0) {
  SetTypeOfTask(GetStaticTypeOfTask());

  GetInput() = in;
  GetOutput() = 0;
}

bool TrofimovNLinearTopologyMPI::ValidationImpl() {
  const auto &in = GetInput();
  return (in.source >= 0 && in.target >= 0 && in.value >= 0);
}

bool TrofimovNLinearTopologyMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

  linear_comm_ = MPI_COMM_WORLD;

  return true;
}

bool TrofimovNLinearTopologyMPI::RunImpl() {
  const auto &in = GetInput();

  if (size_ == 1) {
    GetOutput() = in.value;
    return true;
  }

  if (in.source < 0 || in.target < 0 || in.source >= size_ || in.target >= size_) {
    return false;
  }

  int result = 0;

  if (in.source == in.target) {
    if (rank_ == in.target) {
      result = in.value;
    }

    MPI_Bcast(&result, 1, MPI_INT, in.target, linear_comm_);
    GetOutput() = result;
    return true;
  }

  const int step = (in.target > in.source) ? 1 : -1;
  int current_value = 0;

  if (rank_ == in.source) {
    current_value = in.value;
    MPI_Send(&current_value, 1, MPI_INT, rank_ + step, 0, linear_comm_);
  }

  const bool forward_direction = (step == 1);
  const int start = in.source + step;
  const int end = in.target;

  for (int i = start; (forward_direction && i <= end) || (!forward_direction && i >= end); i += step) {
    if (rank_ == i) {
      MPI_Recv(&current_value, 1, MPI_INT, rank_ - step, 0, linear_comm_, MPI_STATUS_IGNORE);

      if (rank_ != in.target) {
        MPI_Send(&current_value, 1, MPI_INT, rank_ + step, 0, linear_comm_);
      } else {
        result = current_value;
      }
    }
  }

  MPI_Bcast(&result, 1, MPI_INT, in.target, linear_comm_);
  GetOutput() = result;

  return true;
}

bool TrofimovNLinearTopologyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace trofimov_n_linear_topology
