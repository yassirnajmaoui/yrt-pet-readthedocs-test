/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/Globals.hpp"

int Globals::num_threads = omp_get_max_threads();

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

void py_setup_globals(pybind11::module& m)
{
	auto c = pybind11::class_<Globals>(m, "Globals");
	c.def_static("set_num_threads", &Globals::set_num_threads);
	c.def_static("get_num_threads", &Globals::get_num_threads);
}

#endif  // if BUILD_PYBIND11
