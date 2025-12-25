#pragma once

#include "task/include/task.hpp"
#include "trofimov_n_mult_matrix_cannon/common/include/common.hpp"

namespace trofimov_n_mult_matrix_cannon {

class TrofimovNMultMatrixCanonMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TrofimovNMultMatrixCanonMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace trofimov_n_mult_matrix_cannon
