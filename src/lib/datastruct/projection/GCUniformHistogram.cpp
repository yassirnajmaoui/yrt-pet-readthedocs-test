/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/GCUniformHistogram.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

void py_setup_gcuniformhistogram(py::module& m)
{
	auto c =
	    py::class_<GCUniformHistogram, Histogram3D>(m, "GCUniformHistogram");
	c.def(py::init<const GCScanner*, float>(), py::arg("scanner"),
	      py::arg("value") = 1.0f);
}
#endif


GCUniformHistogram::GCUniformHistogram(const GCScanner* p_scanner,
                                       float p_value)
    : Histogram3D(p_scanner), m_value(p_value)
{
}

void GCUniformHistogram::writeToFile(const std::string& filename) const
{
	std::ofstream file;
	file.open(filename.c_str(), std::ios::binary | std::ios::out);
	if (!file.is_open())
	{
		throw std::filesystem::filesystem_error(
		    "The file given \"" + filename + "\" could not be opened",
		    std::make_error_code(std::errc::io_error));
	}
	int magic = MAGIC_NUMBER;
	int num_dims = 3;
	size_t shape[3]{Histogram3D::n_z_bin, Histogram3D::n_phi,
	                Histogram3D::n_r};
	file.write((char*)&magic, sizeof(int));
	file.write((char*)&num_dims, sizeof(int));
	file.write((char*)shape, 3 * sizeof(size_t));

	for (size_t i = 0; i < Histogram3D::count(); i++)
	{
		file.write((char*)&m_value, sizeof(float));
	}
}

float GCUniformHistogram::getProjectionValue(bin_t binId) const
{
	(void)binId;
	return m_value;
}

void GCUniformHistogram::setProjectionValue(bin_t binId, float val)
{
	(void)binId;
	(void)val;
}

void GCUniformHistogram::incrementProjection(bin_t binId, float val)
{
	(void)binId;
	(void)val;
}

void GCUniformHistogram::clearProjections(float p_value)
{
	setValue(p_value);
}

void GCUniformHistogram::setValue(float p_value)
{
	m_value = p_value;
}

bool GCUniformHistogram::isUniform() const
{
	return true;
}
