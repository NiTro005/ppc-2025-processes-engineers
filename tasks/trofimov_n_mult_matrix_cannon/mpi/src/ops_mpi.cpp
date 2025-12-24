#include "trofimov_n_mult_matrix_cannon/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
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
  const auto &[_, __, n] = GetInput();
  if (n > 0) {
    GetOutput().assign(n * n, 0.0);
  }
  return true;
}

bool TrofimovNMultMatrixCanonMPI::RunImpl() {
  const auto &[A, B, n] = GetInput();
  auto &C = GetOutput();

  if (n <= 0) {
    return true;
  }

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int q = static_cast<int>(std::sqrt(world_size));
  if (q * q != world_size || n % q != 0) {
    // fallback: последовательное умножение
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
          C[i * n + j] += A[i * n + k] * B[k * n + j];
        }
      }
    }
    return true;
  }

  int block = n / q;

  int dims[2] = {q, q};
  int periods[2] = {1, 1};
  MPI_Comm cart;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart);

  int cart_rank;
  MPI_Comm_rank(cart, &cart_rank);

  int coords[2];
  MPI_Cart_coords(cart, cart_rank, 2, coords);
  int row = coords[0];
  int col = coords[1];

  std::vector<double> Ablock(block * block);
  std::vector<double> Bblock(block * block);
  std::vector<double> Cblock(block * block, 0.0);

  // === Рассылка блоков ===
  if (cart_rank == 0) {
    for (int p = 0; p < world_size; p++) {
      int pc[2];
      MPI_Cart_coords(cart, p, 2, pc);

      std::vector<double> Ab(block * block);
      std::vector<double> Bb(block * block);

      for (int i = 0; i < block; i++) {
        for (int j = 0; j < block; j++) {
          int gi = pc[0] * block + i;
          int gj = pc[1] * block + j;
          Ab[i * block + j] = A[gi * n + gj];
          Bb[i * block + j] = B[gi * n + gj];
        }
      }

      if (p == 0) {
        Ablock = Ab;
        Bblock = Bb;
      } else {
        MPI_Send(Ab.data(), block * block, MPI_DOUBLE, p, 0, cart);
        MPI_Send(Bb.data(), block * block, MPI_DOUBLE, p, 1, cart);
      }
    }
  } else {
    MPI_Recv(Ablock.data(), block * block, MPI_DOUBLE, 0, 0, cart, MPI_STATUS_IGNORE);
    MPI_Recv(Bblock.data(), block * block, MPI_DOUBLE, 0, 1, cart, MPI_STATUS_IGNORE);
  }

  int left, right, up, down;

  // === Начальное выравнивание ===
  for (int i = 0; i < row; i++) {
    MPI_Cart_shift(cart, 1, -1, &right, &left);
    MPI_Sendrecv_replace(Ablock.data(), block * block, MPI_DOUBLE, left, 0, right, 0, cart, MPI_STATUS_IGNORE);
  }

  for (int i = 0; i < col; i++) {
    MPI_Cart_shift(cart, 0, -1, &down, &up);
    MPI_Sendrecv_replace(Bblock.data(), block * block, MPI_DOUBLE, up, 1, down, 1, cart, MPI_STATUS_IGNORE);
  }

  // === Основной цикл Кэннона ===
  for (int step = 0; step < q; step++) {
    for (int i = 0; i < block; i++) {
      for (int j = 0; j < block; j++) {
        for (int k = 0; k < block; k++) {
          Cblock[i * block + j] += Ablock[i * block + k] * Bblock[k * block + j];
        }
      }
    }

    MPI_Cart_shift(cart, 1, -1, &right, &left);
    MPI_Sendrecv_replace(Ablock.data(), block * block, MPI_DOUBLE, left, 0, right, 0, cart, MPI_STATUS_IGNORE);

    MPI_Cart_shift(cart, 0, -1, &down, &up);
    MPI_Sendrecv_replace(Bblock.data(), block * block, MPI_DOUBLE, up, 1, down, 1, cart, MPI_STATUS_IGNORE);
  }

  // === СБОР ВСЕХ БЛОКОВ НА КАЖДОМ ПРОЦЕССЕ ===
  std::vector<double> all_blocks(world_size * block * block);
  MPI_Allgather(Cblock.data(), block * block, MPI_DOUBLE, all_blocks.data(), block * block, MPI_DOUBLE, cart);

  // === Восстановление полной матрицы C ===
  for (int p = 0; p < world_size; p++) {
    int pc[2];
    MPI_Cart_coords(cart, p, 2, pc);

    const double *src = &all_blocks[p * block * block];

    for (int i = 0; i < block; i++) {
      for (int j = 0; j < block; j++) {
        int gi = pc[0] * block + i;
        int gj = pc[1] * block + j;
        C[gi * n + gj] = src[i * block + j];
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
