/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/PluginFramework.hpp"

#if BUILD_PYBIND11

#include "recon/GCVariable.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void py_setup_array(py::module&);
void py_setup_vector3dall(py::module&);
void py_setup_straightlineparam(py::module&);
void py_setup_tubeofresponse(py::module& m);
void py_setup_timeofflight(py::module& m);

void py_setup_imagebase(py::module&);
void py_setup_imageparams(py::module&);
void py_setup_image(py::module&);
void py_setup_projectiondata(py::module& m);
void py_setup_biniterator(py::module& m);
void py_setup_histogram(py::module& m);
void py_setup_histogram3d(py::module& m);
void py_setup_uniformhistogram(py::module& m);
void py_setup_sparsehistogram(py::module& m);
void py_setup_listmode(py::module& m);
void py_setup_listmodelut(py::module& m);
void py_setup_listmodelutdoi(py::module& m);
void py_setup_projectionlist(py::module& m);
void py_setup_gcdetectorsetup(py::module& m);
void py_setup_gcosem(py::module& m);
void py_setup_gcreconstructionutils(py::module& m);
void py_setup_gcscanner(py::module& m);
void py_setup_gcdetcoord(py::module& m);
void py_setup_gcdetregular(py::module& m);
void py_setup_gcio(py::module& m);

void py_setup_srtm(py::module& m);
void py_setup_imagewarpertemplate(py::module& m);
void py_setup_imagewarpermatrix(py::module& m);
void py_setup_imagewarperfunction(py::module& m);
void py_setup_gcutilities(py::module& m);

void py_setup_operator(py::module& m);
void py_setup_operatorpsf(py::module& m);
void py_setup_operatorprojectorparams(py::module& m);
void py_setup_operatorprojectorbase(py::module& m);
void py_setup_operatorprojector(py::module& m);
void py_setup_operatorprojectorsiddon(py::module& m);
void py_setup_operatorprojectordd(py::module& m);

void py_setup_Globals(py::module& m);

void py_setup_gccrystal(py::module& m);
void py_setup_gcsinglescattersimulator(py::module& m);
void py_setup_gcscatterestimator(py::module& m);

#ifdef BUILD_CUDA
void py_setup_gcimagedevice(py::module&);
void py_setup_gcprojectiondatadevice(py::module& m);
void py_setup_operatorprojectordevice(py::module& m);
void py_setup_operatorprojectordd_gpu(py::module& m);
#endif


PYBIND11_MODULE(pyyrtpet, m)
{
	// GCVariable is added here because the class is empty
	auto gcvariable = py::class_<GCVariable>(m, "GCVariable");

	py_setup_array(m);
	py_setup_vector3dall(m);
	py_setup_straightlineparam(m);
	py_setup_tubeofresponse(m);
	py_setup_timeofflight(m);

	py_setup_imagebase(m);
	py_setup_imageparams(m);
	py_setup_image(m);
	py_setup_biniterator(m);
	py_setup_projectiondata(m);
	py_setup_histogram(m);
	py_setup_histogram3d(m);
	py_setup_uniformhistogram(m);
	py_setup_sparsehistogram(m);
	py_setup_listmode(m);
	py_setup_listmodelut(m);
	py_setup_listmodelutdoi(m);
	py_setup_projectionlist(m);
	py_setup_gcdetectorsetup(m);
	py_setup_gcscanner(m);
	py_setup_gcdetcoord(m);
	py_setup_gcdetregular(m);
	py_setup_gcio(m);

	py_setup_srtm(m);
	py_setup_imagewarpertemplate(m);
	py_setup_imagewarpermatrix(m);
	py_setup_imagewarperfunction(m);
	py_setup_gcutilities(m);

	py_setup_operator(m);
	py_setup_operatorpsf(m);
	py_setup_operatorprojectorbase(m);
	py_setup_operatorprojector(m);
	py_setup_operatorprojectorparams(m);
	py_setup_operatorprojectorsiddon(m);
	py_setup_operatorprojectordd(m);
	py_setup_gcosem(m);
	py_setup_gcreconstructionutils(m);

	py_setup_Globals(m);

	py_setup_gccrystal(m);
	py_setup_gcsinglescattersimulator(m);
	py_setup_gcscatterestimator(m);

#ifdef BUILD_CUDA
	py_setup_gcimagedevice(m);
	py_setup_gcprojectiondatadevice(m);
	py_setup_operatorprojectordevice(m);
	py_setup_operatorprojectordd_gpu(m);
#endif

	// Add the plugins
	Plugin::PluginRegistry::instance().addAllPybind11Modules(m);
}

#endif  // if BUILD_PYBIND11
