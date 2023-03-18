#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "solvers/lcs_factory_franka_new.h"

#include "drake/multibody/plant/multibody_plant.h"

namespace py = pybind11;

namespace dairlib {
namespace pydairlib {

using solvers::LCSFactoryFrankaNew;

PYBIND11_MODULE(lcs_factory_franka_new, m) {
  m.doc() = "Binding lcs factories for c3 lcs modeling";

  using py_rvp = py::return_value_policy;

//  py::class_<LCSFactoryFrankaNew>(m, "LCSFactoryFrankaNew")
//      .def("LinearizePlantToLCS", &LCSFactoryFrankaNew::LinearizePlantToLCS,
//           py::arg("plant"), py::arg("context"), py::arg("plant_ad"), py::arg("context_ad"), py::arg("contact_geoms_orig"),
//           py::arg("num_friction_directions"), py::arg("mu"), py::arg("dt"));
    m.def("LinearizePlantToLCS", &LCSFactoryFrankaNew::LinearizePlantToLCS,
           py::arg("plant"), py::arg("context"), py::arg("plant_ad"), py::arg("context_ad"), py::arg("contact_geoms_orig"),
           py::arg("num_friction_directions"), py::arg("mu"), py::arg("dt"));
}
}  // namespace pydairlib
}  // namespace dairlib