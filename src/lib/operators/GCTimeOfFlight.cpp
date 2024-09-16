/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/GCTimeOfFlight.hpp"

#include "utils/GCTools.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

void py_setup_gctimeofflight(py::module& m)
{
	auto c = py::class_<GCTimeOfFlightHelper>(m, "GCTimeOfFlightHelper");
	c.def(py::init<float, int>());
	c.def("getAlphaRange", &GCTimeOfFlightHelper::getAlphaRange);
	c.def("getWeight", &GCTimeOfFlightHelper::getWeight);
	c.def("getSigma", &GCTimeOfFlightHelper::getSigma);
	c.def("getTruncWidth", &GCTimeOfFlightHelper::getTruncWidth);
	c.def("getNorm", &GCTimeOfFlightHelper::getNorm);
}
#endif

GCTimeOfFlightHelper::GCTimeOfFlightHelper(float tof_width_ps, int tof_n_std)
{
	const double tof_width_mm = tof_width_ps * SPEED_OF_LIGHT_MM_PS * 0.5;
	// FWHM = sigma 2 sqrt(2 ln 2)
	m_sigma = tof_width_mm / (2 * sqrtf(2 * logf(2)));
	if (tof_n_std <= 0)
	{
		m_truncWidth_mm = -1.f;
	}
	else
	{
		m_truncWidth_mm = tof_n_std * m_sigma;
	}
	m_norm = 1 / (std::sqrt(2 * PI) * m_sigma);
}

float GCTimeOfFlightHelper::getSigma() const
{
	return m_sigma;
}

float GCTimeOfFlightHelper::getTruncWidth() const
{
	return m_truncWidth_mm;
}

float GCTimeOfFlightHelper::getNorm() const
{
	return m_norm;
}
