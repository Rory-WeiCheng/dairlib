#pragma once

#include "epos/Definitions.h"
#include "epos/epos_util.h"
#include "labjack/labjack_utils.h"

#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/framework/context.h"
#include "drake/multibody/plant/multibody_plant.h"

#include "systems/framework/timestamped_vector.h"
#include "systems/framework/output_vector.h"
#include "systems/filters/internal_low_pass_filter.h"

using drake::multibody::MultibodyPlant;
using drake::systems::Context;

using dairlib::systems::OutputVector;
using dairlib::InternalLowPassFilter;

using std::string;


namespace dairlib {
class CartpoleOutputInterface : public drake::systems::LeafSystem<double> {
 public:
  CartpoleOutputInterface(const MultibodyPlant<double>& plant);

  void SetupOutputInterface();

 private:
  void ConfigureEpos();
  void ConfigureLabjack();
  void CopyOutput(const drake::systems::Context<double>& context,
              drake::systems::BasicVector<double>* output) const;

  drake::systems::EventStatus DiscreteUpdate(
      const drake::systems::Context<double>& context,
      drake::systems::DiscreteValues<double>* values) const;

  MAXON_HANDLE MotorHandle_ = nullptr;
  LABJACK_HANDLE EncoderHandle_ = nullptr;
  const MultibodyPlant<double>& plant_;

  int prev_x_idx_;
  int effort_idx_;

  int nq_;
  int nv_;

  mutable InternalLowPassFilter vel_filt_ =
      InternalLowPassFilter(0.01, 2);
};
}