#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "trofimov_n_max_val_matrix/common/include/common.hpp"
#include "trofimov_n_max_val_matrix/mpi/include/ops_mpi.hpp"
#include "trofimov_n_max_val_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace trofimov_n_max_val_matrix {

class TrofimovNRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int matrix_size = std::get<0>(params);

    input_data_.clear();
    expected_output_.clear();

    input_data_.reserve(static_cast<std::size_t>(matrix_size));
    expected_output_.reserve(static_cast<std::size_t>(matrix_size));

    for (int i = 0; i < matrix_size; ++i) {
      std::vector<int> row;
      row.reserve(static_cast<std::size_t>(matrix_size));

      for (int j = 0; j < matrix_size; ++j) {
        row.push_back((i * matrix_size) + j);
      }
      input_data_.push_back(row);

      expected_output_.push_back(*std::ranges::max_element(row));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (std::size_t i = 0; i < expected_output_.size(); ++i) {
      if (output_data[i] != expected_output_[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

}  // namespace trofimov_n_max_val_matrix
