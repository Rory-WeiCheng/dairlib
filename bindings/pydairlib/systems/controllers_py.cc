#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "systems/controllers/c3_controller.h"

#include "drake/multibody/plant/multibody_plant.h"

namespace py = pybind11;

namespace dairlib {
namespace pydairlib {

using systems::controllers::C3Controller;

PYBIND11_MODULE(controllers, m) {
  m.doc() = "Binding controller factories for Cassie";

  using py_rvp = py::return_value_policy;

  py::class_<C3Controller, drake::systems::LeafSystem<double>>(m,
                                                               "C3Controller")
      .def(py::init<const drake::multibody::MultibodyPlant<double>&,
                    drake::multibody::MultibodyPlant<double>&,
                    drake::systems::Context<double>&,
                    drake::systems::Context<double>&,
                    const drake::multibody::MultibodyPlant<drake::AutoDiffXd>&,
                    drake::multibody::MultibodyPlant<drake::AutoDiffXd>&,
                    drake::systems::Context<drake::AutoDiffXd>&,
                    drake::systems::Context<drake::AutoDiffXd>&,
                    const drake::geometry::SceneGraph<double>&,
                    const drake::systems::Diagram<double>&,
                    std::vector<drake::geometry::GeometryId>, int, double,
                    const std::vector<Eigen::MatrixXd>&,
                    const std::vector<Eigen::MatrixXd>&,
                    const std::vector<Eigen::MatrixXd>&,
                    const std::vector<Eigen::MatrixXd>&,
                    const std::vector<Eigen::VectorXd>&>(),
           py::arg("plant"), py::arg("plant_f"), py::arg("context"), py::arg("context_f"), py::arg("plant_ad"), py::arg("plant_ad_f"),
           py::arg("context_ad"),  py::arg("context_ad_f"), py::arg("scene_graph"), py::arg("diagram"),  py::arg("contact_geoms"),
           py::arg("num_friction_directions"), py::arg("mu"), py::arg("Q"),
           py::arg("R"), py::arg("G"), py::arg("U"), py::arg("xdesired"))
      .def("get_input_port_config", &C3Controller::get_input_port_config,
           py_rvp::reference_internal)
      .def("get_input_port_output", &C3Controller::get_input_port_output,
           py_rvp::reference_internal);
}
}  // namespace pydairlib
}  // namespace dairlib