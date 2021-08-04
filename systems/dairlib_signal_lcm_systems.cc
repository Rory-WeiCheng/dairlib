#include "systems/dairlib_signal_lcm_systems.h"

#include <limits>
typedef std::numeric_limits<double> dbl;

using std::cout;
using std::endl;

namespace dairlib {
namespace systems {

using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::LeafSystem;
using std::string;

/*--------------------------------------------------------------------------*/
// methods implementation for DairlibSignalReceiver.

DairlibSignalReceiver::DairlibSignalReceiver(int signal_size)
    : signal_size_(signal_size) {
  this->DeclareAbstractInputPort("lcmt_dairlib_signal",
                                 drake::Value<dairlib::lcmt_dairlib_signal>{});
  this->DeclareVectorOutputPort(TimestampedVector<double>(signal_size),
                                &DairlibSignalReceiver::UnpackLcmIntoVector);
}

void DairlibSignalReceiver::UnpackLcmIntoVector(
    const Context<double>& context, TimestampedVector<double>* output) const {
  const drake::AbstractValue* input = this->EvalAbstractInput(context, 0);
  DRAKE_ASSERT(input != nullptr);
  const auto& input_msg = input->get_value<dairlib::lcmt_dairlib_signal>();
  for (int i = 0; i < signal_size_; i++) {
    // We assume that the order of the vector is [data, timestamp]
    output->get_mutable_value()(i) = input_msg.val[i];
  }
  output->set_timestamp(input_msg.utime * 1e-6);
}

/*--------------------------------------------------------------------------*/
// methods implementation for DairlibSignalSender.

DairlibSignalSender::DairlibSignalSender(
    const std::vector<std::string>& signal_names)
    : signal_names_(signal_names),
      signal_size_(signal_names.size()),
      with_hacks_(false) {
  this->DeclareVectorInputPort(BasicVector<double>(signal_names.size()));
  this->DeclareAbstractOutputPort(&DairlibSignalSender::PackVectorIntoLcm);
}

DairlibSignalSender::DairlibSignalSender(
    const std::vector<std::string>& signal_names, double stride_period)
    : signal_names_(signal_names),
      signal_size_(signal_names.size()),
      stride_period_(stride_period),
      with_hacks_(true) {
  this->DeclareVectorInputPort(BasicVector<double>(signal_names.size()));
  this->DeclareAbstractOutputPort(&DairlibSignalSender::PackVectorIntoLcm);
}

void DairlibSignalSender::PackVectorIntoLcm(
    const Context<double>& context, dairlib::lcmt_dairlib_signal* msg) const {
  const auto* input_vector = this->EvalVectorInput(context, 0);

  msg->dim = signal_size_;
  msg->val.resize(signal_size_);
  msg->coord.resize(signal_size_);
  for (int i = 0; i < signal_size_; i++) {
    msg->val[i] = input_vector->get_value()(i);
    msg->coord[i] = signal_names_[i];
  }

  if (with_hacks_) {
    // Hacks for initial fsm state (need -1 for neutral point swing foot, and
    // need 0 for planner)
    if (msg->val[0] == -1) {
      msg->val[0] = 0;
    }

    // Add epsilon to avoid error from converting double to int.
    msg->utime = (context.get_time() + 1e-12) * 1e6;

    // Testing -- Calc phase
    double lift_off_time = input_vector->get_value()(1);
    double time_in_first_mode = (msg->utime * 1e-6) - lift_off_time;
    double init_phase = time_in_first_mode / stride_period_;

    /*cout << "init_phase = " << init_phase<<"\n";
    cout << "fsm state = " << input_vector->get_value()(0) << endl;
    cout << "lift_off_time = " << lift_off_time << endl;
    cout << "current_time = " << context.get_time() << endl;
    cout << "time_in_first_mode = " << time_in_first_mode << endl;
    cout << "input_vector->get_value() = " << input_vector->get_value() << endl;
     */

    if (init_phase > 1) {
      cout.precision(dbl::max_digits10);

      cout << "WARNING: phase = " << init_phase
           << " (>= 1). There might be a bug somewhere, "
              "since we are using a time-based fsm\n";
      cout << "fsm state = " << input_vector->get_value()(0) << endl;
      cout << "lift_off_time = " << lift_off_time << endl;
      cout << "current_time = " << context.get_time() << endl;
      cout << "time_in_first_mode = " << time_in_first_mode << endl;
      cout << "input_vector->get_value() = " << input_vector->get_value()
           << endl;
      cout << '\a';  // making noise to notify
      cout << "======================\n";
      cout << "======================\n";
      cout << "======================\n";
      cout << "======================\n";
      cout << "======================\n";
      //    DRAKE_UNREACHABLE();
    } else if (init_phase < 0) {
      cout.precision(dbl::max_digits10);

      cout << "WARNING: phase = " << init_phase
           << " (<0). There might be a bug somewhere, "
              "since we are using a time-based fsm\n";
      cout << "fsm state = " << input_vector->get_value()(0) << endl;
      cout << "lift_off_time = " << lift_off_time << endl;
      cout << "current_time = " << context.get_time() << endl;
      cout << "time_in_first_mode = " << time_in_first_mode << endl;
      cout << "input_vector->get_value() = " << input_vector->get_value()
           << endl;
      cout << '\a';  // making noise to notify
      cout << "======================\n";
      cout << "======================\n";
      cout << "======================\n";
      cout << "======================\n";
      cout << "======================\n";
      //    DRAKE_UNREACHABLE();
    }
  }
}

/*--------------------------------------------------------------------------*/
// methods implementation for TimestampedVectorSender.

TimestampedVectorSender::TimestampedVectorSender(int signal_size)
    : signal_size_(signal_size) {
  this->DeclareVectorInputPort(BasicVector<double>(signal_size));
  this->DeclareAbstractOutputPort(&TimestampedVectorSender::PackVectorIntoLcm);
}

void TimestampedVectorSender::PackVectorIntoLcm(
    const Context<double>& context,
    dairlib::lcmt_timestamped_vector* msg) const {
  const auto* input_vector = this->EvalVectorInput(context, 0);

  // using the time from the context
  msg->utime = context.get_time() * 1e6;

  msg->size = signal_size_;
  msg->data.resize(signal_size_);
  for (int i = 0; i < signal_size_; i++) {
    msg->data[i] = input_vector->get_value()(i);
  }
}

}  // namespace systems
}  // namespace dairlib