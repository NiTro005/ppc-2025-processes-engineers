#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace trofimov_n_linear_topology {

struct InputData {
  int source;
  int target;
  int value;
};

using InType = InputData;
using OutType = int;

using TestType = std::tuple<InputData, std::string>;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace trofimov_n_linear_topology
