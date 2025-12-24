#include "trofimov_n_mult_matrix_cannon/seq/include/ops_seq.hpp"

namespace trofimov_n_mult_matrix_cannon {

TrofimovNMultMatrixCanonSEQ::TrofimovNMultMatrixCanonSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TrofimovNMultMatrixCanonSEQ::ValidationImpl() {
  return true;
}

bool TrofimovNMultMatrixCanonSEQ::PreProcessingImpl() {
  const auto &[_, __, n] = GetInput();
  if (n > 0) {
    GetOutput().assign(n * n, 0.0);
  }
  return true;
}

bool TrofimovNMultMatrixCanonSEQ::RunImpl() {
  const auto &[A, B, n] = GetInput();
  auto &C = GetOutput();

  if (n <= 0) {
    return true;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
  return true;
}

bool TrofimovNMultMatrixCanonSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace trofimov_n_mult_matrix_cannon
