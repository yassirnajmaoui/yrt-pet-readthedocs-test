/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/GCTubeOfResponse.hpp"
#include "geometry/GCCylinder.hpp"

#include <cmath>
#include <cstdlib>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <sstream>
namespace py = pybind11;

void py_setup_gctubeofresponse(py::module& m)
{
	auto c = py::class_<GCTubeOfResponse>(m, "GCTubeOfResponse");
	c.def(py::init<const GCVector&, const GCVector&, const GCVector&,
	               const GCVector&, float, float>(),
	      py::arg("p1"), py::arg("p2"), py::arg("n1"), py::arg("n2"),
	      py::arg("crystalSize_trans"), py::arg("crystalSize_z"));
	c.def("setBackLine", &GCTubeOfResponse::setBackLine);
	c.def("setFrontLine", &GCTubeOfResponse::setFrontLine);
	c.def("setRightLine", &GCTubeOfResponse::setRightLine);
	c.def("setLeftLine", &GCTubeOfResponse::setLeftLine);
	c.def("getAverageLine", &GCTubeOfResponse::getAverageLine);

	c.def("getLeftLine", &GCTubeOfResponse::getLeftLine);
	c.def("getRightLine", &GCTubeOfResponse::getRightLine);
	c.def("getFrontLine", &GCTubeOfResponse::getFrontLine);
	c.def("getBackLine", &GCTubeOfResponse::getBackLine);
	c.def("getAvgLine", &GCTubeOfResponse::getAvgLine);
	c.def_readwrite("thickness_z", &GCTubeOfResponse::thickness_z);
	c.def_readwrite("thickness_trans", &GCTubeOfResponse::thickness_trans);
	c.def_readwrite("isMoreHorizontalThanVertical",
	                &GCTubeOfResponse::isMoreHorizontalThanVertical);

	c.def("__repr__",
	      [](const GCTubeOfResponse& self)
	      {
		      std::stringstream ss;
		      ss << self;
		      return ss.str();
	      });
}

#endif


const GCStraightLineParam& GCTubeOfResponse::getAverageLine() const
{
	return avgLine;
}

const GCStraightLineParam& GCTubeOfResponse::getAvgLine() const
{
	return getAverageLine();
}

void GCTubeOfResponse::updateAvgLine()
{
	GCVector average_crystal1 =
	    GCVector((leftLine.point1.x + rightLine.point1.x + frontLine.point1.x +
	              backLine.point1.x) /
	                 4.0,
	             (leftLine.point1.y + rightLine.point1.y + frontLine.point1.y +
	              backLine.point1.y) /
	                 4.0,
	             (leftLine.point1.z + rightLine.point1.z + frontLine.point1.z +
	              backLine.point1.z) /
	                 4.0);

	GCVector average_crystal2 =
	    GCVector((leftLine.point2.x + rightLine.point2.x + frontLine.point2.x +
	              backLine.point2.x) /
	                 4.0,
	             (leftLine.point2.y + rightLine.point2.y + frontLine.point2.y +
	              backLine.point2.y) /
	                 4.0,
	             (leftLine.point2.z + rightLine.point2.z + frontLine.point2.z +
	              backLine.point2.z) /
	                 4.0);

	avgLine = GCStraightLineParam(average_crystal1, average_crystal2);
}

std::ostream& operator<<(std::ostream& oss, const GCTubeOfResponse& self)
{
	oss << "Left line: " << self.getLeftLine()
	    << "\nRight line: " << self.getRightLine()
	    << "\nFront line: " << self.getFrontLine()
	    << "\nBack line: " << self.getBackLine()
	    << "\nAverage line: " << self.getAverageLine();
	return oss;
}

GCTubeOfResponse::GCTubeOfResponse(const GCVector& p1, const GCVector& p2,
                                   const GCVector& n1, const GCVector& n2,
                                   float p_thickness_trans, float p_thickness_z)
{
	thickness_z = p_thickness_z;
	thickness_trans = p_thickness_trans;
	avgLine = GCStraightLineParam(p1, p2);
	m_n1 = n1;
	m_n2 = n2;

	GCVector zVect(0.0, 0.0, 1.0);
	GCVector sidesVect1 = n1.crossProduct(zVect);
	GCVector sidesVect2 = n2.crossProduct(zVect);

	GCVector crystal1Left = avgLine.point1 + sidesVect1 * thickness_trans / 2;
	GCVector crystal1Right = avgLine.point1 - sidesVect1 * thickness_trans / 2;
	GCVector crystal1Front = avgLine.point1 - zVect * thickness_z / 2;
	GCVector crystal1Back = avgLine.point1 + zVect * thickness_z / 2;

	sidesVect2 = sidesVect2 * (-1.0);

	GCVector crystal2Left = avgLine.point2 + sidesVect2 * thickness_trans / 2;
	GCVector crystal2Right = avgLine.point2 - sidesVect2 * thickness_trans / 2;
	GCVector crystal2Front = avgLine.point2 - zVect * thickness_z / 2;
	GCVector crystal2Back = avgLine.point2 + zVect * thickness_z / 2;

	leftLine = GCStraightLineParam(crystal1Left, crystal2Left);
	rightLine = GCStraightLineParam(crystal1Right, crystal2Right);
	frontLine = GCStraightLineParam(crystal1Front, crystal2Front);
	backLine = GCStraightLineParam(crystal1Back, crystal2Back);

	GCVector tubevector = avgLine.point2 - avgLine.point1;
	isMoreHorizontalThanVertical =
	    (std::abs(tubevector.y) < std::abs(tubevector.x));
}

bool GCTubeOfResponse::clip(const GCCylinder& cyl)
{
	bool allIntersect = true;
	allIntersect &= cyl.clip_line(&leftLine);
	allIntersect &= cyl.clip_line(&rightLine);
	allIntersect &= cyl.clip_line(&backLine);
	allIntersect &= cyl.clip_line(&frontLine);
	updateAvgLine();
	return allIntersect;
}
