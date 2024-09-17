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
void py_setup_gcvector(py::module&);
void py_setup_gcstraightlineparam(py::module&);
void py_setup_gctubeofresponse(py::module& m);
void py_setup_gctimeofflight(py::module& m);

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

void py_setup_gcoperator(py::module& m);
void py_setup_gcoperatorpsf(py::module& m);
void py_setup_gcoperatorprojectorparams(py::module& m);
void py_setup_gcoperatorprojectorbase(py::module& m);
void py_setup_gcoperatorprojector(py::module& m);
void py_setup_gcoperatorprojectorsiddon(py::module& m);
void py_setup_gcoperatorprojectordd(py::module& m);

void py_setup_gcglobals(py::module& m);

void py_setup_gccrystal(py::module& m);
void py_setup_gcsinglescattersimulator(py::module& m);
void py_setup_gcscatterestimator(py::module& m);

#ifdef BUILD_CUDA
void py_setup_gcimagedevice(py::module&);
void py_setup_gcprojectiondatadevice(py::module& m);
void py_setup_gcoperatorprojectordevice(py::module& m);
void py_setup_gcoperatorprojectordd_gpu(py::module& m);
#endif


PYBIND11_MODULE(pyyrtpet, m)
{
	// GCVariable is added here because the class is empty
	auto gcvariable = py::class_<GCVariable>(m, "GCVariable");

	py_setup_array(m);
	py_setup_gcvector(m);
	py_setup_gcstraightlineparam(m);
	py_setup_gctubeofresponse(m);
	py_setup_gctimeofflight(m);

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

	py_setup_gcoperator(m);
	py_setup_gcoperatorpsf(m);
	py_setup_gcoperatorprojectorbase(m);
	py_setup_gcoperatorprojector(m);
	py_setup_gcoperatorprojectorparams(m);
	py_setup_gcoperatorprojectorsiddon(m);
	py_setup_gcoperatorprojectordd(m);
	py_setup_gcosem(m);
	py_setup_gcreconstructionutils(m);

	py_setup_gcglobals(m);

	py_setup_gccrystal(m);
	py_setup_gcsinglescattersimulator(m);
	py_setup_gcscatterestimator(m);

#ifdef BUILD_CUDA
	py_setup_gcimagedevice(m);
	py_setup_gcprojectiondatadevice(m);
	py_setup_gcoperatorprojectordevice(m);
	py_setup_gcoperatorprojectordd_gpu(m);
#endif

	// Add the plugins
	Plugin::PluginRegistry::instance().addAllPybind11Modules(m);
}

#endif  // if BUILD_PYBIND11
