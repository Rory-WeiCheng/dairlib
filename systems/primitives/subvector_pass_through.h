#pragma once

#include <memory>

#include "drake/common/drake_copyable.h"
#include "drake/systems/framework/leaf_system.h"

namespace dairlib {
namespace systems {

using drake::systems::LeafSystem;
using drake::systems::OutputPort;
using drake::systems::InputPortDescriptor;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::optional;

/// This class is copied from drake/systems/primitives/PassThrough
/// with the modification that it only passes through a subset of the vector
template <typename T>
class SubvectorPassThrough final : public LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SubvectorPassThrough)

  /// Constructs a pass through system (`y = u.segment(start,length)`).
  /// @param vector_size the length of the input vector
  /// @param start the initial index of the subvector
  /// @param length number of elements in the subvector
  explicit SubvectorPassThrough(int vector_size, int start, int length);

  /// Scalar-type converting copy constructor.
  /// See @ref system_scalar_conversion.
  template <typename U>
  explicit SubvectorPassThrough(const SubvectorPassThrough<U>&);

  virtual ~SubvectorPassThrough() {}

  /// Returns the sole input port.
  const InputPortDescriptor<T>& get_input_port() const {
    return LeafSystem<T>::get_input_port(0);
  }

  // Don't use the indexed get_input_port when calling this system directly.
  void get_input_port(int) = delete;

  /// Returns the sole output port.
  const OutputPort<T>& get_output_port() const {
    return LeafSystem<T>::get_output_port(0);
  }

  // Don't use the indexed get_output_port when calling this system directly.
  void get_output_port(int) = delete;

 protected:
  void DoCalcVectorOutput(
      const Context<T>& context,
      BasicVector<T>* output) const;

  // Override feedthrough detection to avoid the need for `DoToSymbolic()`.
  optional<bool> DoHasDirectFeedthrough(
      int input_port, int output_port) const override;

 private:
  bool is_abstract() const { return false;}


  // Allow different specializations to access each other's private data.
  template <typename U> friend class SubvectorPassThrough;

  int start_;
  int length_;
};

}  // namespace systems
}  // namespace dairlib
