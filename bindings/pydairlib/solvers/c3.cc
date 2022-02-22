#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "solvers/c3_miqp.h"

namespace py = pybind11;

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using py::arg;

namespace dairlib {
namespace pydairlib {

using solvers::C3MIQP;


PYBIND11_MODULE(c3, m)
{
	py::class_<C3MIQP> c3miqp(m, "C3MIQP");

	// Bind the two constructors
	// the py::arg arguments aren't strictly necessary, but they allow the python
	// code to use the C3MQP(A: xxx, B: xxx) style
	c3miqp.def(py::init<const vector<MatrixXd>&, const vector<MatrixXd>&,
						     const vector<MatrixXd>&, const vector<MatrixXd>&,
						 		 const vector<MatrixXd>&, const vector<MatrixXd>&,
						     const vector<MatrixXd>&, const vector<VectorXd>&,
						     const vector<MatrixXd>&, const vector<MatrixXd>&,
						     const vector<MatrixXd>&, const C3Options&>(),
						     arg("A"), arg("B"),  arg("D"), arg("d"), arg("E"), arg("F"),
						     arg("H"), arg("c"), arg("Q"), arg("R"), arg("G"),
						     arg("options"));

	c3miqp.def(py::init<const MatrixXd&, const MatrixXd&, const MatrixXd&,
						 	 	 const MatrixXd&, const MatrixXd&, const MatrixXd&,
						 	 	 const MatrixXd&, const VectorXd&, const MatrixXd&,
						 	 	 const MatrixXd&, const MatrixXd&, int, const C3Options&>(),
						 	 	 arg("A"), arg("B"),  arg("D"), arg("d"), arg("E"), arg("F"),
						 	 	 arg("H"), arg("c"), arg("Q"), arg("R"), arg("G"), arg("N"),
						 	 	 arg("options"));

  // An example of binding a simple function. pybind will automatically
  // deduce the arguments, but providing the names here for usability
  c3miqp.def("SolveSingleProjection",
  					 &C3MIQP::SolveSingleProjection, arg("U"), arg("delta_c"), arg("E"),
  					 arg("F"), arg("H"), arg("c"));

	// For binding Solve, because it has multiple return arguments
	// (via pointer, in C++) pointer, the binding is a bit more complex
  c3miqp.def("Solve",
  		[](C3MIQP* self, VectorXd& x0, vector<VectorXd>* delta,
  			     vector<VectorXd>* w) {
  			VectorXd u = self->Solve(x0, delta, w);
  			// Return a tuple of (u, delta, w), by value
  			return py::make_tuple(u, *delta, *w);
  		}
  		);

	py::class_<C3Options> options(m, "C3Options");
	options.def(py::init<>());
	options.def_readwrite("admm_iter", &C3Options::admm_iter);
	options.def_readwrite("rho", &C3Options::rho);
	options.def_readwrite("rho_scale", &C3Options::rho_scale);


}

}	// namespace pydairlib
} // namespace dairlib

