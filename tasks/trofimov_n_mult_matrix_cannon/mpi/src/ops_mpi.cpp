#include "trofimov_n_mult_matrix_cannon/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace trofimov_n_mult_matrix_cannon {

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

  int q = static_cast<int>(std::sqrt(world_size));
  if ((q * q != world_size) || (matrix_size % q != 0)) {
    for (int i = 0; i < matrix_size; i++) {
      for (int j = 0; j < matrix_size; j++) {
        for (int k = 0; k < matrix_size; k++) {
          result_matrix[(i * matrix_size) + j] += matrix_a[(i * matrix_size) + k] * matrix_b[(k * matrix_size) + j];
        }
      }
    }
    return true;
  }

  const int block = matrix_size / q;

  const std::array<int, 2> dims = {q, q};
  const std::array<int, 2> periods = {1, 1};
  MPI_Comm cart = MPI_COMM_NULL;  // Используем MPI_COMM_NULL вместо nullptr
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods.data(), 1, &cart);

  int cart_rank = 0;
  MPI_Comm_rank(cart, &cart_rank);

  std::array<int, 2> coords = {0, 0};
  MPI_Cart_coords(cart, cart_rank, 2, coords.data());
  const int row = coords[0];
  const int col = coords[1];

  const std::size_t block_size = static_cast<std::size_t>(block) * static_cast<std::size_t>(block);
  std::vector<double> block_a(block_size);
  std::vector<double> block_b(block_size);
  std::vector<double> block_c(block_size, 0.0);

  if (cart_rank == 0) {
    for (int process = 0; process < world_size; process++) {
      std::array<int, 2> process_coords = {0, 0};
      MPI_Cart_coords(cart, process, 2, process_coords.data());

      std::vector<double> local_a(block_size);
      std::vector<double> local_b(block_size);

      for (int i = 0; i < block; i++) {
        for (int j = 0; j < block; j++) {
          const int global_i = (process_coords[0] * block) + i;
          const int global_j = (process_coords[1] * block) + j;
          local_a[(i * block) + j] = matrix_a[(global_i * matrix_size) + global_j];
          local_b[(i * block) + j] = matrix_b[(global_i * matrix_size) + global_j];
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

  const std::size_t total_blocks_size = static_cast<std::size_t>(world_size) * block_size;
  std::vector<double> all_blocks(total_blocks_size);
  MPI_Allgather(block_c.data(), static_cast<int>(block_size), MPI_DOUBLE, all_blocks.data(),
                static_cast<int>(block_size), MPI_DOUBLE, cart);

  for (int process = 0; process < world_size; process++) {
    std::array<int, 2> process_coords = {0, 0};
    MPI_Cart_coords(cart, process, 2, process_coords.data());

    const std::size_t offset = static_cast<std::size_t>(process) * block_size;
    const double *src = &all_blocks[offset];

    for (int i = 0; i < block; i++) {
      for (int j = 0; j < block; j++) {
        const int global_i = (process_coords[0] * block) + i;
        const int global_j = (process_coords[1] * block) + j;
        result_matrix[(global_i * matrix_size) + global_j] = src[(i * block) + j];
      }
    }
  }

  MPI_Comm_free(&cart);
  return true;
}

bool TrofimovNMultMatrixCanonMPI::PostProcessingImpl() {
  return true;
}

}  // namespace trofimov_n_mult_matrix_cannon
