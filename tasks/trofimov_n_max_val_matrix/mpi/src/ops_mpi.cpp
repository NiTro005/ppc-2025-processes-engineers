#include "trofimov_n_max_val_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "trofimov_n_max_val_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

namespace trofimov_n_max_val_matrix {

TrofimovNMaxValMatrixMPI::TrofimovNMaxValMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool TrofimovNMaxValMatrixMPI::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }

  size_t cols = GetInput()[0].size();
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

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int rows = input.size();

  int rows_per_process = rows / size;
  int remainder = rows % size;

  int start_row = rank * rows_per_process + std::min(rank, remainder);
  int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);
  int local_rows = end_row - start_row;

  std::vector<int> local_maxima(local_rows);
  for (int i = 0; i < local_rows; i++) {
    int global_row_index = start_row + i;
    if (!input[global_row_index].empty()) {
      local_maxima[i] = *std::max_element(input[global_row_index].begin(), input[global_row_index].end());
    } else {
      local_maxima[i] = 0;
    }
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

  if (rank == 0) {
    GetOutput().resize(rows);
  } else {
    GetOutput().resize(rows);
  }

  MPI_Gatherv(local_maxima.data(), local_rows, MPI_INT, GetOutput().data(), recv_counts.data(), displacements.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(GetOutput().data(), rows, MPI_INT, 0, MPI_COMM_WORLD);

  return rank == 0 ? !GetOutput().empty() : true;
}

bool TrofimovNMaxValMatrixMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace trofimov_n_max_val_matrix
