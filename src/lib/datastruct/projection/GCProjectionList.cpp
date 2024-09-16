/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/GCProjectionList.hpp"

#include "utils/GCAssert.hpp"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void py_setup_gcprojectionlist(pybind11::module& m)
{
	auto c = pybind11::class_<GCProjectionList, IProjectionData>(
	    m, "GCProjectionList", py::buffer_protocol());
	c.def_buffer(
	    [](GCProjectionList& self) -> py::buffer_info
	    {
		    Array1DBase<float>* d = self.getProjectionsArrayRef();
		    return py::buffer_info(d->getRawPointer(), sizeof(float),
		                           py::format_descriptor<float>::format(), 1,
		                           d->getDims(), d->getStrides());
	    });

	auto c_owned = pybind11::class_<GCProjectionListOwned, GCProjectionList>(
	    m, "GCProjectionListOwned");
	c_owned.def(pybind11::init<IProjectionData*>(), py::arg("projectionData"));
	c_owned.def("allocate", &GCProjectionListOwned::allocate);

	auto c_alias = pybind11::class_<GCProjectionListAlias, GCProjectionList>(
	    m, "GCProjectionListAlias");
	c_alias.def(pybind11::init<IProjectionData*>(), py::arg("projectionData"));
	c_alias.def(
	    "Bind",
	    [](GCProjectionListAlias& self, pybind11::buffer& projs_in)
	    {
		    pybind11::buffer_info buffer = projs_in.request();
		    if (buffer.ndim != 1)
		    {
			    throw std::runtime_error(
			        "The buffer given has the wrong dimensions");
		    }
		    if (buffer.format != py::format_descriptor<float>::format())
		    {
			    throw std::invalid_argument(
			        "The buffer given has to have a float32 format");
		    }
		    if (buffer.size !=
		        static_cast<ssize_t>(self.getReference()->count()))
		    {
			    throw std::invalid_argument("The buffer shape does not match "
			                                "with the projection data count "
			                                "count()");
		    }
		    reinterpret_cast<Array1DAlias<float>*>(
		        self.getProjectionsArrayRef())
		        ->bind(reinterpret_cast<float*>(buffer.ptr), buffer.size);
	    });
}

#endif  // if BUILD_PYBIND11

GCProjectionList::GCProjectionList(const IProjectionData* r)
    : IProjectionData(), mp_reference(r)
{
	ASSERT(mp_reference != nullptr);
}

float GCProjectionList::getProjectionValue(bin_t id) const
{
	return (*mp_projs)[id];
}

void GCProjectionList::setProjectionValue(bin_t id, float val)
{
	(*mp_projs)[id] = val;
}

void GCProjectionList::clearProjections(float value)
{
	mp_projs->fill(value);
}

frame_t GCProjectionList::getFrame(bin_t id) const
{
	return mp_reference->getFrame(id);
}

timestamp_t GCProjectionList::getTimestamp(bin_t id) const
{
	return mp_reference->getTimestamp(id);
}

size_t GCProjectionList::getNumFrames() const
{
	return mp_reference->getNumFrames();
}

bool GCProjectionList::isUniform() const
{
	return false;
}

float GCProjectionList::getRandomsEstimate(bin_t id) const
{
	return mp_reference->getRandomsEstimate(id);
}

det_id_t GCProjectionList::getDetector1(bin_t evId) const
{
	return mp_reference->getDetector1(evId);
}

det_id_t GCProjectionList::getDetector2(bin_t evId) const
{
	return mp_reference->getDetector2(evId);
}

det_pair_t GCProjectionList::getDetectorPair(bin_t evId) const
{
	return mp_reference->getDetectorPair(evId);
}

bool GCProjectionList::hasTOF() const
{
	return mp_reference->hasTOF();
}

float GCProjectionList::getTOFValue(bin_t id) const
{
	return mp_reference->getTOFValue(id);
}

bool GCProjectionList::hasMotion() const
{
	return mp_reference->hasMotion();
}

transform_t GCProjectionList::getTransformOfFrame(frame_t frame) const
{
	return mp_reference->getTransformOfFrame(frame);
}

bool GCProjectionList::hasArbitraryLORs() const
{
	return mp_reference->hasArbitraryLORs();
}

line_t GCProjectionList::getArbitraryLOR(bin_t id) const
{
	return mp_reference->getArbitraryLOR(id);
}

Array1DBase<float>* GCProjectionList::getProjectionsArrayRef() const
{
	return (mp_projs.get());
}

size_t GCProjectionList::count() const
{
	return mp_projs->getSize(0);
}

histo_bin_t GCProjectionList::getHistogramBin(bin_t id) const
{
	return mp_reference->getHistogramBin(id);
}

GCProjectionListOwned::GCProjectionListOwned(IProjectionData* r)
    : GCProjectionList(r)
{
	mp_projs = std::make_unique<Array1D<float>>();
}

void GCProjectionListOwned::allocate()
{
	size_t num_events = mp_reference->count();
	std::cout << "Allocating projection list memory"
	          << " for " << num_events << " events" << std::endl;
	static_cast<Array1D<float>*>(mp_projs.get())->allocate(num_events);
	std::cout << "Memory successfully allocated: " << std::flush
	          << mp_projs->getSize(0) << std::endl;
}

GCProjectionListAlias::GCProjectionListAlias(IProjectionData* p)
    : GCProjectionList(p)
{
	mp_projs = std::make_unique<Array1DAlias<float>>();
}

void GCProjectionListAlias::Bind(Array1DBase<float>* projs_in)
{
	static_cast<Array1DAlias<float>*>(mp_projs.get())->bind(*projs_in);
	if (mp_projs->getRawPointer() == nullptr)
	{
		throw std::runtime_error(
		    "An error occured during the ProjectionList binding");
	}
}

std::unique_ptr<BinIterator> GCProjectionList::getBinIter(int numSubsets,
                                                            int idxSubset) const
{
	return mp_reference->getBinIter(numSubsets, idxSubset);
}
