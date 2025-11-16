#include "trofimov_n_max_val_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "trofimov_n_max_val_matrix/common/include/common.hpp"

namespace trofimov_n_max_val_matrix {

TrofimovNMaxValMatrixSEQ::TrofimovNMaxValMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool TrofimovNMaxValMatrixSEQ::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }

  std::size_t cols = GetInput()[0].size();
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
  if (GetInput().empty()) {
    return false;
  }

  for (std::size_t i = 0; i < GetInput().size(); i++) {
    if (!GetInput()[i].empty()) {
      GetOutput()[i] = *std::ranges::max_element(GetInput()[i]);
    }
  }

  return !GetOutput().empty();
}

bool TrofimovNMaxValMatrixSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace trofimov_n_max_val_matrix