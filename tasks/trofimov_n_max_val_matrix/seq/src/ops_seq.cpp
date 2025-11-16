#include "trofimov_n_max_val_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "trofimov_n_max_val_matrix/common/include/common.hpp"

namespace trofimov_n_max_val_matrix {

TrofimovNMaxValMatrixSEQ::TrofimovNMaxValMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in.begin(), in.end());
  GetOutput() = OutType();
}

bool TrofimovNMaxValMatrixSEQ::ValidationImpl() {
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

bool TrofimovNMaxValMatrixSEQ::PreProcessingImpl() {
  GetOutput() = std::vector<int>(GetInput().size(), 0);
  return !GetOutput().empty();
}

bool TrofimovNMaxValMatrixSEQ::RunImpl() {
  const auto &input = GetInput();

  for (std::size_t i = 0; i < input.size(); ++i) {
    if (!input[i].empty()) {
      GetOutput()[i] = *std::ranges::max_element(input[i]);
    }
  }

  return true;
}

bool TrofimovNMaxValMatrixSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace trofimov_n_max_val_matrix
