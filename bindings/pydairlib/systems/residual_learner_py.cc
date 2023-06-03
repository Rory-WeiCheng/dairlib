#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "systems/controllers/residual_learner.h"
#include "solvers/lcs.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace py = pybind11;

namespace dairlib {
namespace pydairlib {

using systems::controllers::ResidualLearner;

PYBIND11_MODULE(residual_learner, m) {
  m.doc() = "Binding Residual Learner";

  using py_rvp = py::return_value_policy;
  py::class_<ResidualLearner, drake::systems::LeafSystem<double>>(m,"ResidualLearner")
      .def(py::init<>())
      .def("CalcResidual", &ResidualLearner::CalcResidual, py::arg("context"), py::arg("residual_lcs"));
}
}  // namespace pydairlib
}  // namespace dairlib