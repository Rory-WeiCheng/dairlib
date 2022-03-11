#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "examples/Cassie/diagrams/osc_running_controller_diagram.h"

#include "drake/multibody/plant/multibody_plant.h"

namespace py = pybind11;

namespace dairlib {
namespace pydairlib {

using examples::controllers::OSCRunningControllerDiagram;

PYBIND11_MODULE(controllers, m) {
  m.doc() = "Binding controller factories for Cassie";

  using py_rvp = py::return_value_policy;

  py::class_<dairlib::examples::controllers::OSCRunningControllerDiagram, drake::systems::Diagram<double>>(
      m, "OSCRunningControllerFactory")
      .def(py::init<const std::string&, const std::string&>(),
           py::arg("osc_gains_filename"), py::arg("osqp_settings_filename"))
      .def("get_state_input_port",
           &OSCRunningControllerDiagram::get_state_input_port,
           py_rvp::reference_internal)
      .def("get_cassie_out_input_port",
           &OSCRunningControllerDiagram::get_cassie_out_input_port,
           py_rvp::reference_internal)
      .def("get_control_output_port",
           &OSCRunningControllerDiagram::get_control_output_port,
           py_rvp::reference_internal)
      .def("get_controller_failure_output_port",
           &OSCRunningControllerDiagram::get_controller_failure_output_port,
           py_rvp::reference_internal);
}
}  // namespace pydairlib
}  // namespace dairlib