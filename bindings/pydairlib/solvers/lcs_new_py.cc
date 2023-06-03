#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "solvers/lcs_new.h"

namespace py = pybind11;
namespace dairlib {
namespace pydairlib {

using solvers::LCSNew;

PYBIND11_MODULE(lcs_new, m) {
  m.doc() = "Binding lcs basic class for lcs_factory_franka_new and let it also return the calculate lambda";
  py::class_<LCSNew>(m, "LCSNew")
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
      .def_readonly("A", &LCSNew::A_)
      .def_readonly("B", &LCSNew::B_)
      .def_readonly("D", &LCSNew::D_)
      .def_readonly("d", &LCSNew::d_)
      .def_readonly("E", &LCSNew::E_)
      .def_readonly("F", &LCSNew::F_)
      .def_readonly("H", &LCSNew::H_)
      .def_readonly("c", &LCSNew::c_)
      .def_readonly("N", &LCSNew::N_)
      .def("Simulate", &LCSNew::Simulate,py::arg("x_init"), py::arg("input"));

}
}  // namespace pydairlib
}  // namespace dairlib