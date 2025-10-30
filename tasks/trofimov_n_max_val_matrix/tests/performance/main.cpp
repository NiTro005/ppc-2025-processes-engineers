#include <gtest/gtest.h>

#include "trofimov_n_max_val_matrix/common/include/common.hpp"
#include "trofimov_n_max_val_matrix/mpi/include/ops_mpi.hpp"
#include "trofimov_n_max_val_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace trofimov_n_max_val_matrix {

class MaxValMatrixRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(MaxValMatrixRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TrofimovNMaxValMatrixMPI, TrofimovNMaxValMatrixSEQ>(PPC_SETTINGS_trofimov_n_max_val_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MaxValMatrixRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MaxValMatrixRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace trofimov_n_max_val_matrix
