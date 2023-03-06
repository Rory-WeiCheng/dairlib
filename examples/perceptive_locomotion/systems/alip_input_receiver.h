#pragma once

#include "multibody/multibody_utils.h"
#include "drake/systems/framework/leaf_system.h"

/// System to translate an incoming lcmt_saved_traj message containing an ankle
/// torque to a vector of desired robot ankle torques
namespace dairlib::perceptive_locomotion {
class AlipInputReceiver : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(AlipInputReceiver);
  AlipInputReceiver(const drake::multibody::MultibodyPlant<double>& plant,
                    std::vector<int> left_right_fsm_states,
                    std::vector<std::string> left_right_ankle_motor_names);

 private:

  void CopyInput(const drake::systems::Context<double>& context,
                 drake::systems::BasicVector<double>* out) const;

  const int nu_;
  drake::systems::InputPortIndex fsm_input_port_;
  drake::systems::InputPortIndex input_traj_input_port_;
  const std::vector<int> left_right_fsm_states_;
  std::unordered_map<int, int> fsm_to_stance_ankle_map_;
};
}