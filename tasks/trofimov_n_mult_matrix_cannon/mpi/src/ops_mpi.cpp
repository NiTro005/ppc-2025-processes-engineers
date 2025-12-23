#include "trofimov_n_mult_matrix_cannon/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "trofimov_n_mult_matrix_cannon/common/include/common.hpp"

namespace trofimov_n_mult_matrix_cannon {

TrofimovNMultMatrixCanonMPI::TrofimovNMultMatrixCanonMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TrofimovNMultMatrixCanonMPI::ValidationImpl() {
  const auto &[A, B, n] = GetInput();
  return n > 0 && A.size() == static_cast<size_t>(n * n) && B.size() == static_cast<size_t>(n * n);
}

bool TrofimovNMultMatrixCanonMPI::PreProcessingImpl() {
  const auto &[_, __, n] = GetInput();
  GetOutput().assign(n * n, 0.0);
  return true;
}

bool TrofimovNMultMatrixCanonMPI::RunImpl() {
  const auto &[A, B, n] = GetInput();
  auto &C = GetOutput();

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int q = static_cast<int>(std::sqrt(size));
  if (q * q != size || n % q != 0) {
    return false;
  }

  int block = n / q;

  int dims[2] = {q, q};
  int periods[2] = {1, 1};
  MPI_Comm cart;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart);

  int coords[2];
  MPI_Cart_coords(cart, rank, 2, coords);
  int row = coords[0];
  int col = coords[1];

  std::vector<double> Ablock(block * block);
  std::vector<double> Bblock(block * block);
  std::vector<double> Cblock(block * block, 0.0);

  /* -------- Scatter blocks -------- */
  if (rank == 0) {
    for (int p = 0; p < size; p++) {
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

  /* -------- Initial Cannon shift -------- */
  int left, right, up, down;

  MPI_Cart_shift(cart, 1, -row, &right, &left);
  MPI_Sendrecv_replace(Ablock.data(), block * block, MPI_DOUBLE, left, 0, right, 0, cart, MPI_STATUS_IGNORE);

  MPI_Cart_shift(cart, 0, -col, &down, &up);
  MPI_Sendrecv_replace(Bblock.data(), block * block, MPI_DOUBLE, up, 1, down, 1, cart, MPI_STATUS_IGNORE);

  /* -------- Cannon iterations -------- */
  for (int step = 0; step < q; step++) {
    for (int i = 0; i < block; i++) {
      for (int j = 0; j < block; j++) {
        double sum = Cblock[i * block + j];
        for (int k = 0; k < block; k++) {
          sum += Ablock[i * block + k] * Bblock[k * block + j];
        }
        Cblock[i * block + j] = sum;
      }
    }

    MPI_Cart_shift(cart, 1, -1, &right, &left);
    MPI_Sendrecv_replace(Ablock.data(), block * block, MPI_DOUBLE, left, 0, right, 0, cart, MPI_STATUS_IGNORE);

    MPI_Cart_shift(cart, 0, -1, &down, &up);
    MPI_Sendrecv_replace(Bblock.data(), block * block, MPI_DOUBLE, up, 1, down, 1, cart, MPI_STATUS_IGNORE);
  }

  MPI_Barrier(cart);

  /* -------- Gather result -------- */
  if (rank == 0) {
    for (int p = 0; p < size; p++) {
      int pc[2];
      MPI_Cart_coords(cart, p, 2, pc);

      std::vector<double> tmp(block * block);
      if (p == 0) {
        tmp = Cblock;
      } else {
        MPI_Recv(tmp.data(), block * block, MPI_DOUBLE, p, 2, cart, MPI_STATUS_IGNORE);
      }

      for (int i = 0; i < block; i++) {
        for (int j = 0; j < block; j++) {
          int gi = pc[0] * block + i;
          int gj = pc[1] * block + j;
          C[gi * n + gj] = tmp[i * block + j];
        }
      }
    }
  } else {
    MPI_Send(Cblock.data(), block * block, MPI_DOUBLE, 0, 2, cart);
  }

  MPI_Comm_free(&cart);
  return true;
}

bool TrofimovNMultMatrixCanonMPI::PostProcessingImpl() {
  return true;
}

}  // namespace trofimov_n_mult_matrix_cannon
