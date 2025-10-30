#pragma once

#include "trofimov_n_max_val_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace trofimov_n_max_val_matrix {

class TrofimovNMaxValMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TrofimovNMaxValMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace trofimov_n_max_val_matrix
