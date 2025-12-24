#include "trofimov_n_mult_matrix_cannon/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "trofimov_n_mult_matrix_cannon/common/include/common.hpp"

namespace trofimov_n_mult_matrix_cannon {

namespace {

void MultiplySequential(const std::vector<double> &a, const std::vector<double> &b, int n,
                        std::vector<double> &result) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        result[(i * n) + j] += a[(i * n) + k] * b[(k * n) + j];
      }
    }
  }
}

MPI_Comm CreateCartesianComm(int q) {
  const std::array<int, 2> dims = {q, q};
  const std::array<int, 2> periods = {1, 1};

  MPI_Comm cart = MPI_COMM_NULL;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods.data(), 1, &cart);
  return cart;
}

void ScatterBlocks(MPI_Comm cart, int block, int world_size, const std::vector<double> &matrix_a,
                   const std::vector<double> &matrix_b, int matrix_size, std::vector<double> &block_a,
                   std::vector<double> &block_b) {
  const std::size_t block_size = static_cast<std::size_t>(block) * static_cast<std::size_t>(block);

  int rank = 0;
  MPI_Comm_rank(cart, &rank);

  if (rank == 0) {
    for (int process = 0; process < world_size; process++) {
      std::array<int, 2> coords{};
      MPI_Cart_coords(cart, process, 2, coords.data());

      std::vector<double> local_a(block_size);
      std::vector<double> local_b(block_size);

      for (int i = 0; i < block; i++) {
        for (int j = 0; j < block; j++) {
          const int gi = (coords[0] * block) + i;
          const int gj = (coords[1] * block) + j;
          local_a[(i * block) + j] = matrix_a[(gi * matrix_size) + gj];
          local_b[(i * block) + j] = matrix_b[(gi * matrix_size) + gj];
        }
      }

      if (process == 0) {
        block_a = std::move(local_a);
        block_b = std::move(local_b);
      } else {
        MPI_Send(local_a.data(), static_cast<int>(block_size), MPI_DOUBLE, process, 0, cart);
        MPI_Send(local_b.data(), static_cast<int>(block_size), MPI_DOUBLE, process, 1, cart);
      }
    }
  } else {
    MPI_Recv(block_a.data(), static_cast<int>(block_size), MPI_DOUBLE, 0, 0, cart, MPI_STATUS_IGNORE);
    MPI_Recv(block_b.data(), static_cast<int>(block_size), MPI_DOUBLE, 0, 1, cart, MPI_STATUS_IGNORE);
  }
}

void InitialShift(MPI_Comm cart, int row, int col, std::vector<double> &block_a, std::vector<double> &block_b,
                  std::size_t block_size) {
  int left = 0;
  int right = 0;
  int up = 0;
  int down = 0;

  for (int i = 0; i < row; i++) {
    MPI_Cart_shift(cart, 1, -1, &right, &left);
    MPI_Sendrecv_replace(block_a.data(), static_cast<int>(block_size), MPI_DOUBLE, left, 0, right, 0, cart,
                         MPI_STATUS_IGNORE);
  }

  for (int i = 0; i < col; i++) {
    MPI_Cart_shift(cart, 0, -1, &down, &up);
    MPI_Sendrecv_replace(block_b.data(), static_cast<int>(block_size), MPI_DOUBLE, up, 1, down, 1, cart,
                         MPI_STATUS_IGNORE);
  }
}

void CannonMultiply(MPI_Comm cart, int q, int block, std::vector<double> &block_a, std::vector<double> &block_b,
                    std::vector<double> &block_c) {
  const std::size_t block_size = static_cast<std::size_t>(block) * static_cast<std::size_t>(block);

  int left = 0;
  int right = 0;
  int up = 0;
  int down = 0;

  for (int step = 0; step < q; step++) {
    for (int i = 0; i < block; i++) {
      for (int j = 0; j < block; j++) {
        for (int k = 0; k < block; k++) {
          block_c[(i * block) + j] += block_a[(i * block) + k] * block_b[(k * block) + j];
        }
      }
    }

    MPI_Cart_shift(cart, 1, -1, &right, &left);
    MPI_Sendrecv_replace(block_a.data(), static_cast<int>(block_size), MPI_DOUBLE, left, 0, right, 0, cart,
                         MPI_STATUS_IGNORE);

    MPI_Cart_shift(cart, 0, -1, &down, &up);
    MPI_Sendrecv_replace(block_b.data(), static_cast<int>(block_size), MPI_DOUBLE, up, 1, down, 1, cart,
                         MPI_STATUS_IGNORE);
  }
}

void GatherResult(MPI_Comm cart, int world_size, int block, int matrix_size, const std::vector<double> &block_c,
                  std::vector<double> &result) {
  const std::size_t block_size = static_cast<std::size_t>(block) * static_cast<std::size_t>(block);

  std::vector<double> all_blocks(static_cast<std::size_t>(world_size) * block_size);

  MPI_Allgather(block_c.data(), static_cast<int>(block_size), MPI_DOUBLE, all_blocks.data(),
                static_cast<int>(block_size), MPI_DOUBLE, cart);

  for (int process = 0; process < world_size; process++) {
    std::array<int, 2> coords{};
    MPI_Cart_coords(cart, process, 2, coords.data());

    const double *src = &all_blocks[static_cast<std::size_t>(process) * block_size];

    for (int i = 0; i < block; i++) {
      for (int j = 0; j < block; j++) {
        const int gi = (coords[0] * block) + i;
        const int gj = (coords[1] * block) + j;
        result[(gi * matrix_size) + gj] = src[(i * block) + j];
      }
    }
  }
}

}  // namespace

TrofimovNMultMatrixCanonMPI::TrofimovNMultMatrixCanonMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TrofimovNMultMatrixCanonMPI::ValidationImpl() {
  return true;
}

bool TrofimovNMultMatrixCanonMPI::PreProcessingImpl() {
  const auto &[a_vector, b_vector, matrix_size] = GetInput();
  if (matrix_size > 0) {
    GetOutput().assign(static_cast<std::size_t>(matrix_size) * static_cast<std::size_t>(matrix_size), 0.0);
  }
  return true;
}

bool TrofimovNMultMatrixCanonMPI::RunImpl() {
  const auto &[matrix_a, matrix_b, matrix_size] = GetInput();
  auto &result_matrix = GetOutput();

  if (matrix_size <= 0) {
    return true;
  }

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int q = static_cast<int>(std::sqrt(world_size));
  if ((q * q != world_size) || (matrix_size % q != 0)) {
    MultiplySequential(matrix_a, matrix_b, matrix_size, result_matrix);
    return true;
  }

  const int block = matrix_size / q;
  MPI_Comm cart = CreateCartesianComm(q);

  int rank = 0;
  MPI_Comm_rank(cart, &rank);

  std::array<int, 2> coords{};
  MPI_Cart_coords(cart, rank, 2, coords.data());

  const std::size_t block_size = static_cast<std::size_t>(block) * static_cast<std::size_t>(block);

  std::vector<double> block_a(block_size);
  std::vector<double> block_b(block_size);
  std::vector<double> block_c(block_size, 0.0);

  ScatterBlocks(cart, block, world_size, matrix_a, matrix_b, matrix_size, block_a, block_b);

  InitialShift(cart, coords[0], coords[1], block_a, block_b, block_size);
  CannonMultiply(cart, q, block, block_a, block_b, block_c);
  GatherResult(cart, world_size, block, matrix_size, block_c, result_matrix);

  MPI_Comm_free(&cart);
  return true;
}

bool TrofimovNMultMatrixCanonMPI::PostProcessingImpl() {
  return true;
}

}  // namespace trofimov_n_mult_matrix_cannon
