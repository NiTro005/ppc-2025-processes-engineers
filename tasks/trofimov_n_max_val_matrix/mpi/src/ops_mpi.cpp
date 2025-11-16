#include "trofimov_n_max_val_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "trofimov_n_max_val_matrix/common/include/common.hpp"

namespace trofimov_n_max_val_matrix {

TrofimovNMaxValMatrixMPI::TrofimovNMaxValMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool TrofimovNMaxValMatrixMPI::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }

  std::size_t cols = GetInput()[0].size();
  for (const auto &row : GetInput()) {
    if (row.size() != cols) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool TrofimovNMaxValMatrixMPI::PreProcessingImpl() {
  GetOutput() = std::vector<int>(GetInput().size(), 0);
  return !GetOutput().empty();
}

bool TrofimovNMaxValMatrixMPI::RunImpl() {
  auto &input = GetInput();
  if (input.empty()) {
    return false;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::size_t rows = input.size();

  const int rows_per_process = static_cast<int>(rows) / size;
  const int remainder = static_cast<int>(rows) % size;

  const int start_row = (rank * rows_per_process) + std::min(rank, remainder);
  const int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);
  const int local_rows = end_row - start_row;

  std::vector<int> local_maxima(local_rows);
  for (int i = 0; i < local_rows; i++) {
    const int global_row_index = start_row + i;
    local_maxima[i] = *std::max_element(input[global_row_index].begin(), input[global_row_index].end());
  }

  std::vector<int> all_maxima;
  if (rank == 0) {
    all_maxima.resize(rows);
  }

  std::vector<int> recv_counts(size);
  std::vector<int> displacements(size);

  MPI_Gather(&local_rows, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    displacements[0] = 0;
    for (int i = 1; i < size; i++) {
      displacements[i] = displacements[i - 1] + recv_counts[i - 1];
    }
  }

  MPI_Gatherv(local_maxima.data(), local_rows, MPI_INT, all_maxima.data(), recv_counts.data(), displacements.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = all_maxima;
  } else {
    GetOutput().clear();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  const int rows_int = static_cast<int>(rows);
  if (rank == 0) {
    MPI_Bcast(GetOutput().data(), rows_int, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    GetOutput().resize(rows);
    MPI_Bcast(GetOutput().data(), rows_int, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return rank == 0 ? !GetOutput().empty() : true;
}

bool TrofimovNMaxValMatrixMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool success = true;

  if (rank == 0) {
    success = !GetOutput().empty();
  }

  int success_int = success ? 1 : 0;
  MPI_Bcast(&success_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  success = (success_int != 0);

  return success;
}

}  // namespace trofimov_n_max_val_matrix
