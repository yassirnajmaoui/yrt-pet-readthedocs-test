/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/IProjectionData.hpp"

#include "utils/GCGlobals.hpp"

#include <stdexcept>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

void py_setup_iprojectiondata(py::module& m)
{
	auto c = py::class_<IProjectionData, GCVariable>(m, "IProjectionData");
	c.def("count", &IProjectionData::count);
	c.def("getProjectionValue", &IProjectionData::getProjectionValue);
	c.def("setProjectionValue", &IProjectionData::setProjectionValue);
	c.def("getFrame", &IProjectionData::getFrame);
	c.def("getDetector1", &IProjectionData::getDetector1);
	c.def("getDetector2", &IProjectionData::getDetector2);
	c.def("getDetectorPair",
	      [](const IProjectionData& self, bin_t ev)
	      {
		      auto [d1, d2] = self.getDetectorPair(ev);
		      return py::make_tuple(d1, d2);
	      });
	c.def("getHistogramBin", &IProjectionData::getHistogramBin);
	c.def("getBinIter", &IProjectionData::getBinIter);
	c.def("isUniform", &IProjectionData::isUniform);
	c.def("hasMotion", &IProjectionData::hasMotion);
	c.def("getNumFrames", &IProjectionData::getNumFrames);
	c.def("getTransformOfFrame",
	      [](const IProjectionData& self, bin_t bin)
	      {
		      transform_t t = self.getTransformOfFrame(bin);
		      // Return the raw data
		      return py::make_tuple(t.r00, t.r01, t.r02, t.r10, t.r11, t.r12,
		                            t.r20, t.r21, t.r22, t.tx, t.ty, t.tz);
	      });
	c.def("hasTOF", &IProjectionData::hasTOF);
	c.def("getTOFValue", &IProjectionData::getTOFValue);
	c.def("getRandomsEstimate", &IProjectionData::getRandomsEstimate);
	c.def("clearProjections", &IProjectionData::clearProjections);
	c.def("hasArbitraryLORs", &IProjectionData::hasArbitraryLORs);
	c.def("getArbitraryLOR",
	      [](const IProjectionData& self, bin_t bin)
	      {
		      line_t l = self.getArbitraryLOR(bin);
		      // Return the raw data
		      return py::make_tuple(l.x1, l.y1, l.z1, l.x2, l.y2, l.z2);
	      });
	c.def("divideMeasurements", &IProjectionData::divideMeasurements);
}

#endif  // if BUILD_PYBIND11

void IProjectionData::operationOnEachBin(
    const std::function<float(bin_t)>& func)
{
	for (bin_t i = 0; i < count(); i++)
	{
		setProjectionValue(i, func(i));
	}
}

void IProjectionData::operationOnEachBinParallel(
    const std::function<float(bin_t)>& func)
{
	int num_threads = GCGlobals::get_num_threads();
	bin_t i;
#pragma omp parallel for num_threads(num_threads) default(none) private(i), \
    firstprivate(func)
	for (i = 0u; i < count(); i++)
	{
		setProjectionValue(i, func(i));
	}
}

bool IProjectionData::isUniform() const
{
	return false;
}

bool IProjectionData::hasMotion() const
{
	return false;
}

size_t IProjectionData::getNumFrames() const
{
	// By default, only one frame
	return 1ull;
}

transform_t IProjectionData::getTransformOfFrame(frame_t frame) const
{
	(void)frame;
	// Return identity rotation and null translation
	return {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
}

float IProjectionData::getTOFValue(bin_t id) const
{
	(void)id;
	throw std::logic_error("getTOFValue unimplemented");
}

float IProjectionData::getRandomsEstimate(bin_t id) const
{
	(void)id;
	return 0.0f;
}

bool IProjectionData::hasTOF() const
{
	return false;
}

bool IProjectionData::hasArbitraryLORs() const
{
	return false;
}

line_t IProjectionData::getArbitraryLOR(bin_t id) const
{
	(void)id;
	throw std::logic_error("getArbitraryLOR Unimplemented");
}

timestamp_t IProjectionData::getTimestamp(bin_t id) const
{
	(void)id;
	return 0u;
}

frame_t IProjectionData::getFrame(bin_t id) const
{
	(void)id;
	return 0u;
}

det_pair_t IProjectionData::getDetectorPair(bin_t id) const
{
	return {getDetector1(id), getDetector2(id)};
}

histo_bin_t IProjectionData::getHistogramBin(bin_t bin) const
{
	return getDetectorPair(bin);
}

void IProjectionData::clearProjections(float value)
{
	(void)value;
	throw std::logic_error("clearProjections undefined on this object");
}

void IProjectionData::divideMeasurements(const IProjectionData* measurements,
                                         const GCBinIterator* binIter)
{
	int num_threads = GCGlobals::get_num_threads();
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
