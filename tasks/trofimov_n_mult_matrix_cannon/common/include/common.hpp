#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace trofimov_n_mult_matrix_cannon {

using Matrix = std::vector<double>;

using InType = std::tuple<Matrix, Matrix, int>;
using OutType = Matrix;

using TestType = std::tuple<InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace trofimov_n_mult_matrix_cannon
