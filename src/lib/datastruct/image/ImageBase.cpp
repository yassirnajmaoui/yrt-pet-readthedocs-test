/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/image/ImageBase.hpp"
#include "geometry/GCConstants.hpp"

#include "nlohmann/json.hpp"
#include <fstream>

using json = nlohmann::json;

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

void py_setup_imagebase(py::module& m)
{
	auto c = py::class_<ImageBase, GCVariable>(m, "ImageBase");

	c.def("setValue", &ImageBase::setValue, py::arg("initValue"));
	c.def("addFirstImageToSecond", &ImageBase::addFirstImageToSecond,
	      py::arg("second"));
	c.def("applyThreshold", &ImageBase::applyThreshold, py::arg("maskImage"),
	      py::arg("threshold"), py::arg("val_le_scale"), py::arg("val_le_off"),
	      py::arg("val_gt_scale"), py::arg("val_gt_off"));
	c.def("writeToFile", &ImageBase::writeToFile, py::arg("filename"));
	c.def("updateEMThreshold", &ImageBase::updateEMThreshold,
	      py::arg("update_img"), py::arg("norm_img"), py::arg("threshold"));
}

void py_setup_imageparams(py::module& m)
{
	auto c = py::class_<ImageParams>(m, "ImageParams");
	c.def(py::init<>());
	c.def(py::init<int, int, int, double, double, double, double, double,
	               double>(),
	      py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("length_x"),
	      py::arg("length_y"), py::arg("length_z"), py::arg("offset_x") = 0.,
	      py::arg("offset_y") = 0., py::arg("offset_z") = 0.);
	c.def(py::init<std::string>());
	c.def(py::init<const ImageParams&>());
	c.def_readwrite("nx", &ImageParams::nx);
	c.def_readwrite("ny", &ImageParams::ny);
	c.def_readwrite("nz", &ImageParams::nz);

	c.def_readwrite("length_x", &ImageParams::length_x);
	c.def_readwrite("length_y", &ImageParams::length_y);
	c.def_readwrite("length_z", &ImageParams::length_z);

	c.def_readwrite("off_x", &ImageParams::off_x);
	c.def_readwrite("off_y", &ImageParams::off_y);
	c.def_readwrite("off_z", &ImageParams::off_z);

	// Automatically populated fields
	c.def_readwrite("vx", &ImageParams::vx);
	c.def_readwrite("vy", &ImageParams::vy);
	c.def_readwrite("vz", &ImageParams::vz);
	c.def_readwrite("fov_radius", &ImageParams::fov_radius);

	c.def("setup", &ImageParams::setup);
	c.def("serialize", &ImageParams::serialize);

	c.def(py::pickle(
	    [](const ImageParams& g)
	    {
		    nlohmann::json geom_json;
		    g.writeToJSON(geom_json);
		    std::stringstream oss;
		    oss << geom_json;
		    return oss.str();
	    },
	    [](const std::string& s)
	    {
		    nlohmann::json geom_json;
		    geom_json = json::parse(s);
		    ImageParams g;
		    g.readFromJSON(geom_json);
		    return g;
	    }));
}
#endif


ImageParams::ImageParams()
    : nx(-1),
      ny(-1),
      nz(-1),
      length_x(-1.0),
      length_y(-1.0),
      length_z(-1.0),
      off_x(0.0),
      off_y(0.0),
      off_z(0.0),
      vx(-1.0),
      vy(-1.0),
      vz(-1.0)
{
}

ImageParams::ImageParams(int nxi, int nyi, int nzi, double length_xi,
                             double length_yi, double length_zi,
                             double offset_xi, double offset_yi,
                             double offset_zi)
    : nx(nxi),
      ny(nyi),
      nz(nzi),
      length_x(length_xi),
      length_y(length_yi),
      length_z(length_zi),
      off_x(offset_xi),
      off_y(offset_yi),
      off_z(offset_zi)
{
	setup();
}

ImageParams::ImageParams(const ImageParams& in)
{
	copy(in);
}

ImageParams& ImageParams::operator=(const ImageParams& in)
{
	copy(in);
	return *this;
}

void ImageParams::copy(const ImageParams& in)
{
	nx = in.nx;
	ny = in.ny;
	nz = in.nz;
	length_x = in.length_x;
	length_y = in.length_y;
	length_z = in.length_z;
	off_x = in.off_x;
	off_y = in.off_y;
	off_z = in.off_z;
	setup();
}

ImageParams::ImageParams(const std::string& fname)
{
	deserialize(fname);
}

void ImageParams::setup()
{
	vx = length_x / (double)nx;
	vy = length_y / (double)ny;
	vz = length_z / (double)nz;
	fov_radius = (float)(std::max(length_x / 2, length_y / 2));
	fov_radius -= (float)(std::max(vx, vy) / 1000);
}

void ImageParams::serialize(const std::string& fname) const
{
	std::ofstream output(fname);
	json geom_json;
	writeToJSON(geom_json);
	output << geom_json << std::endl;
}

void ImageParams::writeToJSON(json& geom_json) const
{
	geom_json["GCIMAGEPARAMS_FILE_VERSION"] = GCIMAGEPARAMS_FILE_VERSION;
	geom_json["nx"] = nx;
	geom_json["ny"] = ny;
	geom_json["nz"] = nz;
	geom_json["length_x"] = length_x;
	geom_json["length_y"] = length_y;
	geom_json["length_z"] = length_z;
	geom_json["off_x"] = off_x;
	geom_json["off_y"] = off_y;
	geom_json["off_z"] = off_z;
}

void ImageParams::deserialize(const std::string& fname)
{
	std::ifstream json_file(fname);
	if (!json_file.is_open())
	{
		throw std::runtime_error("Error opening ImageParams file: " + fname);
	}
	json geom_json;
	json_file >> geom_json;
	readFromJSON(geom_json);
}

void ImageParams::readFromJSON(json& geom_json)
{
	if (geom_json["GCIMAGEPARAMS_FILE_VERSION"] != GCIMAGEPARAMS_FILE_VERSION)
	{
		throw std::logic_error("Error in ImageParams file version");
	}
	nx = geom_json["nx"].get<int>();
	ny = geom_json["ny"].get<int>();
	nz = geom_json["nz"].get<int>();
	// TODO: Add possibility to provide "vx"
	length_x = geom_json["length_x"].get<double>();
	length_y = geom_json["length_y"].get<double>();
	length_z = geom_json["length_z"].get<double>();
	off_x = geom_json.value("off_x", 0.0);
	off_y = geom_json.value("off_y", 0.0);
	off_z = geom_json.value("off_z", 0.0);
	setup();
}

bool ImageParams::isValid() const
{
	return nx > 0 && ny > 0 && nz > 0 && length_x > 0. && length_y > 0. &&
	       length_z > 0.;
}

bool ImageParams::isSameDimensionsAs(const ImageParams& other) const
{
	return nx == other.nx && ny == other.ny && nz == other.nz;
}

bool ImageParams::isSameLengthsAs(const ImageParams& other) const
{
	return std::abs(length_x - other.length_x) < SMALL &&
	       std::abs(length_z - other.length_z) < SMALL &&
	       std::abs(length_z - other.length_z) < SMALL;
}

bool ImageParams::isSameOffsetsAs(const ImageParams& other) const
{
	return std::abs(off_x - other.off_x) < SMALL &&
	       std::abs(off_z - other.off_z) < SMALL &&
	       std::abs(off_z - other.off_z) < SMALL;
}

bool ImageParams::isSameAs(const ImageParams& other) const
{
	return isSameDimensionsAs(other) && isSameLengthsAs(other) &&
	       isSameOffsetsAs(other);
}

ImageBase::ImageBase(const ImageParams& img_params) : m_params(img_params)
{
}

const ImageParams& ImageBase::getParams() const
{
	return m_params;
}

void ImageBase::setParams(const ImageParams& newParams)
{
	m_params = newParams;
}

float ImageBase::getRadius() const
{
	return m_params.fov_radius;
}
