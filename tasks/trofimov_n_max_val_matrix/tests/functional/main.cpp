#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "trofimov_n_max_val_matrix/common/include/common.hpp"
#include "trofimov_n_max_val_matrix/mpi/include/ops_mpi.hpp"
#include "trofimov_n_max_val_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

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
    
    for (int i = 0; i < matrix_size; ++i) {
      std::vector<int> row;
      for (int j = 0; j < matrix_size; ++j) {
        row.push_back(i * matrix_size + j);
      }
      input_data_.push_back(row);
      
      expected_output_.push_back(*std::max_element(row.begin(), row.end()));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    
    for (size_t i = 0; i < expected_output_.size(); ++i) {
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

namespace {

TEST_P(TrofimovNRunFuncTestsProcesses, MaxValMatrixTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
    std::make_tuple(2, "2x2_matrix"),
    std::make_tuple(3, "3x3_matrix"), 
    std::make_tuple(4, "4x4_matrix")
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<TrofimovNMaxValMatrixMPI, InType>(kTestParam, PPC_SETTINGS_trofimov_n_max_val_matrix),
                   ppc::util::AddFuncTask<TrofimovNMaxValMatrixSEQ, InType>(kTestParam, PPC_SETTINGS_trofimov_n_max_val_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TrofimovNRunFuncTestsProcesses::PrintFuncTestName<TrofimovNRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MaxValMatrixTests, TrofimovNRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace trofimov_n_max_val_matrix