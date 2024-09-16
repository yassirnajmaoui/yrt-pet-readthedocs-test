/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/scanner/GCDetectorSetup.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;
void py_setup_gcdetectorsetup(pybind11::module& m)
{
	auto c = py::class_<GCDetectorSetup>(m, "GCDetectorSetup");
	c.def("getNumDets", &GCDetectorSetup::getNumDets);
	c.def("getXpos", &GCDetectorSetup::getXpos);
	c.def("getYpos", &GCDetectorSetup::getYpos);
	c.def("getZpos", &GCDetectorSetup::getZpos);
	c.def("getXorient", &GCDetectorSetup::getXorient);
	c.def("getYorient", &GCDetectorSetup::getYorient);
	c.def("getZorient", &GCDetectorSetup::getZorient);
	c.def("getPos", &GCDetectorSetup::getPos);
	c.def("getOrient", &GCDetectorSetup::getOrient);
	c.def("writeToFile", &GCDetectorSetup::writeToFile);
}
#endif


GCVector GCDetectorSetup::getPos(det_id_t id) const
{
	return {getXpos(id), getYpos(id), getZpos(id)};
}

GCVector GCDetectorSetup::getOrient(det_id_t id) const
{
	return {getXorient(id), getYorient(id), getZorient(id)};
}
