/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/GCGlobals.hpp"

int GCGlobals::num_threads = omp_get_max_threads();

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

void py_setup_gcglobals(pybind11::module& m)
{
	auto c = pybind11::class_<GCGlobals>(m, "GCGlobals");
	c.def_static("set_num_threads", &GCGlobals::set_num_threads);
	c.def_static("get_num_threads", &GCGlobals::get_num_threads);
}

#endif  // if BUILD_PYBIND11
