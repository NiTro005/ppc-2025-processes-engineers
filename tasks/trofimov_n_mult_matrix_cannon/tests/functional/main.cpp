#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "trofimov_n_mult_matrix_cannon/common/include/common.hpp"
#include "trofimov_n_mult_matrix_cannon/mpi/include/ops_mpi.hpp"
#include "trofimov_n_mult_matrix_cannon/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace trofimov_n_mult_matrix_cannon {

class TrofimovNFuncTestsMultMatrixCanon : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    return std::get<1>(param);
  }

 protected:
  void SetUp() override {
    const auto &param = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const std::string &name = std::get<1>(param);

    if (name == "n1") {
      n_ = 1;
      A_ = {2.0};
      B_ = {3.0};
    } else if (name == "n2_identity") {
      n_ = 2;
      A_ = {1.0, 0.0, 0.0, 1.0};
      B_ = {5.0, 6.0, 7.0, 8.0};
    } else if (name == "n2_zero") {
      n_ = 2;
      A_ = {0.0, 0.0, 0.0, 0.0};
      B_ = {1.0, 2.0, 3.0, 4.0};
    } else if (name == "n4") {
      n_ = 4;
      A_.resize(static_cast<Matrix::size_type>(n_) * n_);
      B_.resize(static_cast<Matrix::size_type>(n_) * n_);
      std::iota(A_.begin(), A_.end(), 1.0);
      std::iota(B_.begin(), B_.end(), -1.0);
    } else if (name == "invalid_n") {
      n_ = 0;
    }

    input_data_ = std::make_tuple(A_, B_, n_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output) final {
    const auto &[A, B, n] = input_data_;

    if (n <= 0 || A.size() != static_cast<size_t>(n) * n || B.size() != static_cast<size_t>(n) * n) {
      return true;
    }

    std::vector<double> expected(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
          sum += A[i * n + k] * B[k * n + j];
        }
        expected[i * n + j] = sum;
      }
    }

    const double eps = 1e-9;
    for (size_t i = 0; i < expected.size(); ++i) {
      if (std::fabs(expected[i] - output[i]) > eps) {
        return false;
      }
    }
    return true;
  }

 private:
  Matrix A_;
  Matrix B_;
  int n_{0};
  InType input_data_;
};

TEST_P(TrofimovNFuncTestsMultMatrixCanon, RunFuncTests) {
  ExecuteTest(GetParam());
}

namespace {

const std::array<TestType, 5> kTestParams = {TestType{InType{}, "n1"}, TestType{InType{}, "n2_identity"},
                                             TestType{InType{}, "n2_zero"}, TestType{InType{}, "n4"},
                                             TestType{InType{}, "invalid_n"}};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<TrofimovNMultMatrixCanonMPI, InType>(
                                               kTestParams, PPC_SETTINGS_trofimov_n_mult_matrix_cannon),
                                           ppc::util::AddFuncTask<TrofimovNMultMatrixCanonSEQ, InType>(
                                               kTestParams, PPC_SETTINGS_trofimov_n_mult_matrix_cannon));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = TrofimovNFuncTestsMultMatrixCanon::PrintFuncTestName<TrofimovNFuncTestsMultMatrixCanon>;

INSTANTIATE_TEST_SUITE_P(MatrixCanonFuncTests, TrofimovNFuncTestsMultMatrixCanon, kGtestValues, kTestName);

}  // namespace

}  // namespace trofimov_n_mult_matrix_cannon
