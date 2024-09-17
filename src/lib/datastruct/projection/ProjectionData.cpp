/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/ProjectionData.hpp"

#include "utils/Globals.hpp"

#include <stdexcept>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

void py_setup_projectiondata(py::module& m)
{
	auto c = py::class_<ProjectionData, GCVariable>(m, "ProjectionData");
	c.def("count", &ProjectionData::count);
	c.def("getProjectionValue", &ProjectionData::getProjectionValue);
	c.def("setProjectionValue", &ProjectionData::setProjectionValue);
	c.def("getFrame", &ProjectionData::getFrame);
	c.def("getDetector1", &ProjectionData::getDetector1);
	c.def("getDetector2", &ProjectionData::getDetector2);
	c.def("getDetectorPair",
	      [](const ProjectionData& self, bin_t ev)
	      {
		      auto [d1, d2] = self.getDetectorPair(ev);
		      return py::make_tuple(d1, d2);
	      });
	c.def("getHistogramBin", &ProjectionData::getHistogramBin);
	c.def("getBinIter", &ProjectionData::getBinIter);
	c.def("isUniform", &ProjectionData::isUniform);
	c.def("hasMotion", &ProjectionData::hasMotion);
	c.def("getNumFrames", &ProjectionData::getNumFrames);
	c.def("getTransformOfFrame",
	      [](const ProjectionData& self, bin_t bin)
	      {
		      transform_t t = self.getTransformOfFrame(bin);
		      // Return the raw data
		      return py::make_tuple(t.r00, t.r01, t.r02, t.r10, t.r11, t.r12,
		                            t.r20, t.r21, t.r22, t.tx, t.ty, t.tz);
	      });
	c.def("hasTOF", &ProjectionData::hasTOF);
	c.def("getTOFValue", &ProjectionData::getTOFValue);
	c.def("getRandomsEstimate", &ProjectionData::getRandomsEstimate);
	c.def("clearProjections", &ProjectionData::clearProjections);
	c.def("hasArbitraryLORs", &ProjectionData::hasArbitraryLORs);
	c.def("getArbitraryLOR",
	      [](const ProjectionData& self, bin_t bin)
	      {
		      line_t l = self.getArbitraryLOR(bin);
		      // Return the raw data
		      return py::make_tuple(l.x1, l.y1, l.z1, l.x2, l.y2, l.z2);
	      });
	c.def("divideMeasurements", &ProjectionData::divideMeasurements);
}

#endif  // if BUILD_PYBIND11

void ProjectionData::operationOnEachBin(
    const std::function<float(bin_t)>& func)
{
	for (bin_t i = 0; i < count(); i++)
	{
		setProjectionValue(i, func(i));
	}
}

void ProjectionData::operationOnEachBinParallel(
    const std::function<float(bin_t)>& func)
{
	int num_threads = Globals::get_num_threads();
	bin_t i;
#pragma omp parallel for num_threads(num_threads) default(none) private(i), \
    firstprivate(func)
	for (i = 0u; i < count(); i++)
	{
		setProjectionValue(i, func(i));
	}
}

bool ProjectionData::isUniform() const
{
	return false;
}

bool ProjectionData::hasMotion() const
{
	return false;
}

size_t ProjectionData::getNumFrames() const
{
	// By default, only one frame
	return 1ull;
}

transform_t ProjectionData::getTransformOfFrame(frame_t frame) const
{
	(void)frame;
	// Return identity rotation and null translation
	return {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
}

float ProjectionData::getTOFValue(bin_t id) const
{
	(void)id;
	throw std::logic_error("getTOFValue unimplemented");
}

float ProjectionData::getRandomsEstimate(bin_t id) const
{
	(void)id;
	return 0.0f;
}

bool ProjectionData::hasTOF() const
{
	return false;
}

bool ProjectionData::hasArbitraryLORs() const
{
	return false;
}

line_t ProjectionData::getArbitraryLOR(bin_t id) const
{
	(void)id;
	throw std::logic_error("getArbitraryLOR Unimplemented");
}

timestamp_t ProjectionData::getTimestamp(bin_t id) const
{
	(void)id;
	return 0u;
}

frame_t ProjectionData::getFrame(bin_t id) const
{
	(void)id;
	return 0u;
}

det_pair_t ProjectionData::getDetectorPair(bin_t id) const
{
	return {getDetector1(id), getDetector2(id)};
}

histo_bin_t ProjectionData::getHistogramBin(bin_t bin) const
{
	return getDetectorPair(bin);
}

void ProjectionData::clearProjections(float value)
{
	(void)value;
	throw std::logic_error("clearProjections undefined on this object");
}

void ProjectionData::divideMeasurements(const ProjectionData* measurements,
                                         const BinIterator* binIter)
{
	int num_threads = Globals::get_num_threads();
#pragma omp parallel for num_threads(num_threads)
	for (size_t binIdx = 0; binIdx < binIter->size(); binIdx++)
	{
		size_t bin = binIter->get(binIdx);
		// to prevent numerical instability
		if (getProjectionValue(bin) > 1e-8)
		{
			setProjectionValue(bin, measurements->getProjectionValue(bin) /
			                            getProjectionValue(bin));
		}
	}
}
