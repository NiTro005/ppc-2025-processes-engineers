#include "trofimov_n_mult_matrix_cannon/seq/include/ops_seq.hpp"

#include <cstddef>

#include "trofimov_n_mult_matrix_cannon/common/include/common.hpp"

namespace trofimov_n_mult_matrix_cannon {

TrofimovNMultMatrixCanonSEQ::TrofimovNMultMatrixCanonSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TrofimovNMultMatrixCanonSEQ::ValidationImpl() {
  return true;
}

bool TrofimovNMultMatrixCanonSEQ::PreProcessingImpl() {
  const auto &[a_vector, b_vector, matrix_size] = GetInput();
  if (matrix_size > 0) {
    GetOutput().assign(static_cast<std::size_t>(matrix_size) * static_cast<std::size_t>(matrix_size), 0.0);
  }
  return true;
}

bool TrofimovNMultMatrixCanonSEQ::RunImpl() {
  const auto &[matrix_a, matrix_b, matrix_size] = GetInput();
  auto &result_matrix = GetOutput();

  if (matrix_size <= 0) {
    return true;
  }

  for (int i = 0; i < matrix_size; i++) {
    for (int j = 0; j < matrix_size; j++) {
      for (int k = 0; k < matrix_size; k++) {
        result_matrix[(i * matrix_size) + j] += matrix_a[(i * matrix_size) + k] * matrix_b[(k * matrix_size) + j];
      }
    }
  }
  return true;
}

bool TrofimovNMultMatrixCanonSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace trofimov_n_mult_matrix_cannon
