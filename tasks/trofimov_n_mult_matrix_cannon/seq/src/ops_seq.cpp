#include "trofimov_n_mult_matrix_cannon/seq/include/ops_seq.hpp"

namespace trofimov_n_mult_matrix_cannon {

TrofimovNMultMatrixCanonSEQ::TrofimovNMultMatrixCanonSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TrofimovNMultMatrixCanonSEQ::ValidationImpl() {
  const auto &[A, B, n] = GetInput();
  return n > 0 && A.size() == static_cast<size_t>(n * n) && B.size() == static_cast<size_t>(n * n);
}

bool TrofimovNMultMatrixCanonSEQ::PreProcessingImpl() {
  const auto &[_, __, n] = GetInput();
  GetOutput().assign(n * n, 0.0);
  return true;
}

bool TrofimovNMultMatrixCanonSEQ::RunImpl() {
  const auto &[A, B, n] = GetInput();
  auto &C = GetOutput();

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) {
        sum += A[i * n + k] * B[k * n + j];
      }
      C[i * n + j] = sum;
    }
  }
  return true;
}

bool TrofimovNMultMatrixCanonSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace trofimov_n_mult_matrix_cannon
