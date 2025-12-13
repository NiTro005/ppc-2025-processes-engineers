#pragma once

#include <mpi.h>

#include "task/include/task.hpp"
#include "trofimov_n_linear_topology/common/include/common.hpp"

namespace trofimov_n_linear_topology {

class TrofimovNLinearTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TrofimovNLinearTopologyMPI(const InType &in);

 private:
  MPI_Comm linear_comm_ = MPI_COMM_NULL;
  int rank_ = 0;
  int size_ = 0;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  bool IsValidSourceTarget(const InType &in) const;
  int HandleSpecialCases(const InType &in);
  int PassValueThroughLinearTopology(const InType &in);
  int GetStepDirection(const InType &in) const;
  int ProcessIntermediateNodes(const InType &in, int step, int current_value);
  bool ShouldContinueLoop(int i, int end, bool forward_direction) const;
};

}  // namespace trofimov_n_linear_topology
