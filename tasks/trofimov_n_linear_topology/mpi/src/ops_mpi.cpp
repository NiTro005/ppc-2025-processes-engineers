#include "trofimov_n_linear_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>

#include "trofimov_n_linear_topology/common/include/common.hpp"

namespace trofimov_n_linear_topology {

namespace {

int GetStepDirection(const InType &in) {
  return (in.target > in.source) ? 1 : -1;
}

bool ShouldContinueLoop(int i, int end, bool forward_direction) {
  return forward_direction ? (i <= end) : (i >= end);
}

bool IsValidSourceTarget(const InType &in, int size) {
  return in.source >= 0 && in.target >= 0 && in.source < size && in.target < size;
}

int ProcessIntermediateNodes(const InType &in, int step, int rank, MPI_Comm linear_comm) {
  const bool forward_direction = (step == 1);
  const int start = in.source + step;
  const int end = in.target;
  int current_value = 0;
  int result = 0;

  for (int i = start; ShouldContinueLoop(i, end, forward_direction); i += step) {
    if (rank == i) {
      MPI_Recv(&current_value, 1, MPI_INT, rank - step, 0, linear_comm, MPI_STATUS_IGNORE);

      if (rank != in.target) {
        MPI_Send(&current_value, 1, MPI_INT, rank + step, 0, linear_comm);
      } else {
        result = current_value;
      }
    }
  }

  return result;
}

int PassValueThroughLinearTopology(const InType &in, int rank, MPI_Comm linear_comm) {
  const int step = GetStepDirection(in);
  int current_value = 0;

  if (rank == in.source) {
    current_value = in.value;
    MPI_Send(&current_value, 1, MPI_INT, rank + step, 0, linear_comm);
  }

  return ProcessIntermediateNodes(in, step, rank, linear_comm);
}

int HandleSpecialCases(const InType &in, int rank, MPI_Comm linear_comm) {
  if (in.source == in.target) {
    int result = 0;
    if (rank == in.target) {
      result = in.value;
    }
    MPI_Bcast(&result, 1, MPI_INT, in.target, linear_comm);
    return result;
  }
  return -1;
}

}  // namespace

TrofimovNLinearTopologyMPI::TrofimovNLinearTopologyMPI(const InType &in) : linear_comm_(MPI_COMM_NULL) {
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

  if (!IsValidSourceTarget(in, size_)) {
    return false;
  }

  int result = HandleSpecialCases(in, rank_, linear_comm_);

  if (result != -1) {
    GetOutput() = result;
    return true;
  }

  result = PassValueThroughLinearTopology(in, rank_, linear_comm_);
  MPI_Bcast(&result, 1, MPI_INT, in.target, linear_comm_);
  GetOutput() = result;

  return true;
}

bool TrofimovNLinearTopologyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace trofimov_n_linear_topology
