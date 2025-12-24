#include <gtest/gtest.h>

#include <numeric>
#include <ranges>
#include <tuple>

#include "trofimov_n_mult_matrix_cannon/common/include/common.hpp"
#include "trofimov_n_mult_matrix_cannon/mpi/include/ops_mpi.hpp"
#include "trofimov_n_mult_matrix_cannon/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace trofimov_n_mult_matrix_cannon {

class TrofimovNPerfTestsMultMatrixCanon : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    constexpr int kN = 128;

    Matrix aMatrix(static_cast<Matrix::size_type>(kN) * kN);
    Matrix bMatrix(static_cast<Matrix::size_type>(kN) * kN);

    std::iota(aMatrix.begin(), aMatrix.end(), 1.0);
    std::iota(bMatrix.begin(), bMatrix.end(), -1.0);

    input_data_ = std::make_tuple(aMatrix, bMatrix, kN);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    (void)output_data;
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(TrofimovNPerfTestsMultMatrixCanon, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TrofimovNMultMatrixCanonMPI, TrofimovNMultMatrixCanonSEQ>(
        PPC_SETTINGS_trofimov_n_mult_matrix_cannon);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TrofimovNPerfTestsMultMatrixCanon::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TrofimovNPerfTestsMultMatrixCanon, kGtestValues, kPerfTestName);

}  // namespace trofimov_n_mult_matrix_cannon
