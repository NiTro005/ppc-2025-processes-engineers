#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "trofimov_n_max_val_matrix/common/include/common.hpp"
#include "trofimov_n_max_val_matrix/mpi/include/ops_mpi.hpp"
#include "trofimov_n_max_val_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace trofimov_n_max_val_matrix {

namespace {
constexpr int kDefaultMatrixSize = 100;
}  // namespace

class MaxValMatrixRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixSize_ = kDefaultMatrixSize;

  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    input_data_.clear();
    expected_output_.clear();

    input_data_.reserve(static_cast<std::size_t>(kMatrixSize_));
    expected_output_.reserve(static_cast<std::size_t>(kMatrixSize_));

    for (int i = 0; i < kMatrixSize_; ++i) {
      std::vector<int> row;
      row.reserve(static_cast<std::size_t>(kMatrixSize_));

      for (int j = 0; j < kMatrixSize_; ++j) {
        row.push_back(i * kMatrixSize_ + j);
      }
      input_data_.push_back(std::move(row));

      expected_output_.push_back(*std::ranges::max_element(input_data_.back()));
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
};

TEST_P(MaxValMatrixRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, TrofimovNMaxValMatrixMPI, TrofimovNMaxValMatrixSEQ>(
    PPC_SETTINGS_trofimov_n_max_val_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MaxValMatrixRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MaxValMatrixRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace trofimov_n_max_val_matrix
