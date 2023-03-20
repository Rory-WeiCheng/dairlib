#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "solvers/lcs.h"

namespace py = pybind11;
namespace dairlib {
namespace pydairlib {

using solvers::LCS;

PYBIND11_MODULE(lcs, m) {
  m.doc() = "Binding lcs basic class for lcs_factory_franka_new";
  py::class_<LCS>(m, "LCS")
      .def(py::init<const std::vector<Eigen::MatrixXd>&,const std::vector<Eigen::MatrixXd>&,
          const std::vector<Eigen::MatrixXd>&,const std::vector<Eigen::VectorXd>&,
          const std::vector<Eigen::MatrixXd>&,const std::vector<Eigen::MatrixXd>&,
          const std::vector<Eigen::MatrixXd>&,const std::vector<Eigen::VectorXd>&>(),
          py::arg("A"), py::arg("B"),py::arg("D"),py::arg("d"),
          py::arg("E"), py::arg("F"),py::arg("H"),py::arg("c"))
      .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
          const Eigen::MatrixXd&, const Eigen::VectorXd&,
          const Eigen::MatrixXd&, const Eigen::MatrixXd&,
          const Eigen::MatrixXd&, const Eigen::VectorXd&, const int&>(),py::arg("A"), py::arg("B"),py::arg("D"),py::arg("d"),
          py::arg("E"), py::arg("F"),py::arg("H"),py::arg("c"),py::arg("N"))
      .def_readonly("A", &LCS::A_)
      .def_readonly("B", &LCS::B_)
      .def_readonly("D", &LCS::D_)
      .def_readonly("d", &LCS::d_)
      .def_readonly("E", &LCS::E_)
      .def_readonly("F", &LCS::F_)
      .def_readonly("H", &LCS::H_)
      .def_readonly("c", &LCS::c_)
      .def_readonly("N", &LCS::N_)
      .def("Simulate", &LCS::Simulate,py::arg("x_init"), py::arg("input"));

}
}  // namespace pydairlib
}  // namespace dairlib