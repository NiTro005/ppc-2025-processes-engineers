#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace trofimov_n_max_val_matrix {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace trofimov_n_max_val_matrix
