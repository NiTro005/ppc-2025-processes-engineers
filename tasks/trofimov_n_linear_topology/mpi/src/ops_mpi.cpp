#include "trofimov_n_linear_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>

#include "trofimov_n_linear_topology/common/include/common.hpp"

namespace trofimov_n_linear_topology {

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

  if (!IsValidSourceTarget(in)) {
    return false;
  }

  int result = HandleSpecialCases(in);

  if (result != -1) {
    GetOutput() = result;
    return true;
  }

  result = PassValueThroughLinearTopology(in);
  MPI_Bcast(&result, 1, MPI_INT, in.target, linear_comm_);
  GetOutput() = result;

  return true;
}

bool TrofimovNLinearTopologyMPI::IsValidSourceTarget(const InType &in) const {
  return !(in.source < 0 || in.target < 0 || in.source >= size_ || in.target >= size_);
}

int TrofimovNLinearTopologyMPI::HandleSpecialCases(const InType &in) {
  if (in.source == in.target) {
    int result = 0;
    if (rank_ == in.target) {
      result = in.value;
    }
    MPI_Bcast(&result, 1, MPI_INT, in.target, linear_comm_);
    return result;
  }
  return -1;  // -1 означает, что это не особый случай
}

int TrofimovNLinearTopologyMPI::PassValueThroughLinearTopology(const InType &in) {
  const int step = GetStepDirection(in);
  int current_value = 0;

  if (rank_ == in.source) {
    current_value = in.value;
    MPI_Send(&current_value, 1, MPI_INT, rank_ + step, 0, linear_comm_);
  }

  return ProcessIntermediateNodes(in, step, current_value);
}

int TrofimovNLinearTopologyMPI::GetStepDirection(const InType &in) const {
  return (in.target > in.source) ? 1 : -1;
}

int TrofimovNLinearTopologyMPI::ProcessIntermediateNodes(const InType &in, int step, int current_value) {
  const bool forward_direction = (step == 1);
  const int start = in.source + step;
  const int end = in.target;
  int result = 0;

  for (int i = start; ShouldContinueLoop(i, end, forward_direction); i += step) {
    if (rank_ == i) {
      MPI_Recv(&current_value, 1, MPI_INT, rank_ - step, 0, linear_comm_, MPI_STATUS_IGNORE);

      if (rank_ != in.target) {
        MPI_Send(&current_value, 1, MPI_INT, rank_ + step, 0, linear_comm_);
      } else {
        result = current_value;
      }
    }
  }

  return result;
}

bool TrofimovNLinearTopologyMPI::ShouldContinueLoop(int i, int end, bool forward_direction) const {
  if (forward_direction) {
    return i <= end;
  }
  return i >= end;
}

bool TrofimovNLinearTopologyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace trofimov_n_linear_topology
