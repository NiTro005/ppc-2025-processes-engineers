#include "trofimov_n_linear_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>

#include "trofimov_n_linear_topology/common/include/common.hpp"

namespace trofimov_n_linear_topology {

TrofimovNLinearTopologyMPI::TrofimovNLinearTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TrofimovNLinearTopologyMPI::ValidationImpl() {
  const auto &in = GetInput();
  return in.source >= 0 && in.target >= 0 && in.value >= 0;
}

bool TrofimovNLinearTopologyMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
  GetOutput() = 0;
  return true;
}

bool TrofimovNLinearTopologyMPI::RunImpl() {
  const auto &in = GetInput();

  if (size_ == 1) {
    Work(in.value);
    GetOutput() = in.value;
    return true;
  }

  Work(in.value / size_ + 1);

  int result = 0;

  if (in.source == in.target) {
    if (rank_ == in.source) {
      result = in.value;
    }
    MPI_Bcast(&result, 1, MPI_INT, in.source, MPI_COMM_WORLD);
    GetOutput() = result;
    return true;
  }

  const int step = (in.target > in.source) ? 1 : -1;

  if (rank_ == in.source) {
    result = in.value;
    MPI_Send(&result, 1, MPI_INT, rank_ + step, 0, MPI_COMM_WORLD);
  }

  for (int r = in.source + step; r != in.target + step; r += step) {
    if (rank_ == r) {
      MPI_Recv(&result, 1, MPI_INT, rank_ - step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (r != in.target) {
        MPI_Send(&result, 1, MPI_INT, rank_ + step, 0, MPI_COMM_WORLD);
      }
    }
  }

  MPI_Bcast(&result, 1, MPI_INT, in.target, MPI_COMM_WORLD);
  GetOutput() = result;

  return true;
}

bool TrofimovNLinearTopologyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace trofimov_n_linear_topology
