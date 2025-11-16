#include "trofimov_n_max_val_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "trofimov_n_max_val_matrix/common/include/common.hpp"

namespace trofimov_n_max_val_matrix {

namespace {
constexpr int kRootRank = 0;
}  // namespace

TrofimovNMaxValMatrixMPI::TrofimovNMaxValMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in.begin(), in.end());
  GetOutput() = OutType();
}

bool TrofimovNMaxValMatrixMPI::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }

  const std::size_t cols = GetInput()[0].size();
  for (const auto &row : GetInput()) {
    if (row.size() != cols) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool TrofimovNMaxValMatrixMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == kRootRank) {
    GetOutput() = std::vector<int>(GetInput().size(), 0);
  } else {
    GetOutput().clear();
  }

  return true;
}

bool TrofimovNMaxValMatrixMPI::RunImpl() {
  const auto &input = GetInput();

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto rows = static_cast<int>(input.size());

  if (rows == 0) {
    if (rank == kRootRank) {
      GetOutput().clear();
    }
    return true;
  }

  const int rows_per_process = rows / size;
  const int remainder = rows % size;

  const int start_row = rank * rows_per_process + std::min(rank, remainder);
  const int local_rows = rows_per_process + (rank < remainder ? 1 : 0);

  if (local_rows <= 0 || start_row >= rows) {
    if (rank == kRootRank) {
      std::vector<int> recv_counts(size, 0);
      std::vector<int> displacements(size, 0);
      MPI_Gather(&local_rows, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, kRootRank, MPI_COMM_WORLD);
      GetOutput().resize(static_cast<std::size_t>(rows));
    } else {
      int zero = 0;
      MPI_Gather(&zero, 1, MPI_INT, nullptr, 0, MPI_INT, kRootRank, MPI_COMM_WORLD);
    }
    return true;
  }

  std::vector<int> local_maxima(static_cast<std::size_t>(local_rows));
  for (int i = 0; i < local_rows; ++i) {
    const int global_row_index = start_row + i;
    if (global_row_index < rows) {
      local_maxima[static_cast<std::size_t>(i)] =
          *std::ranges::max_element(input[static_cast<std::size_t>(global_row_index)]);
    }
  }

  std::vector<int> recv_counts(static_cast<std::size_t>(size), 0);
  std::vector<int> displacements(static_cast<std::size_t>(size), 0);

  MPI_Gather(&local_rows, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, kRootRank, MPI_COMM_WORLD);

  if (rank == kRootRank) {
    displacements[0] = 0;
    for (int i = 1; i < size; ++i) {
      displacements[static_cast<std::size_t>(i)] =
          displacements[static_cast<std::size_t>(i - 1)] + recv_counts[static_cast<std::size_t>(i - 1)];
    }

    if (GetOutput().size() != static_cast<std::size_t>(rows)) {
      GetOutput().resize(static_cast<std::size_t>(rows));
    }
  }

  MPI_Gatherv(local_maxima.data(), local_rows, MPI_INT, rank == kRootRank ? GetOutput().data() : nullptr,
              recv_counts.data(), displacements.data(), MPI_INT, kRootRank, MPI_COMM_WORLD);

  return true;
}

bool TrofimovNMaxValMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace trofimov_n_max_val_matrix
