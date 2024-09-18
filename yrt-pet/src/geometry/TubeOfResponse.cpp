/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/TubeOfResponse.hpp"
#include "geometry/Cylinder.hpp"

#include <cmath>
#include <cstdlib>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <sstream>
namespace py = pybind11;

void py_setup_tubeofresponse(py::module& m)
{
	auto c = py::class_<TubeOfResponse>(m, "TubeOfResponse");
	c.def(py::init<const Vector3D&, const Vector3D&, const Vector3D&,
	               const Vector3D&, float, float>(),
	      py::arg("p1"), py::arg("p2"), py::arg("n1"), py::arg("n2"),
	      py::arg("crystalSize_trans"), py::arg("crystalSize_z"));
	c.def("setBackLine", &TubeOfResponse::setBackLine);
	c.def("setFrontLine", &TubeOfResponse::setFrontLine);
	c.def("setRightLine", &TubeOfResponse::setRightLine);
	c.def("setLeftLine", &TubeOfResponse::setLeftLine);
	c.def("getAverageLine", &TubeOfResponse::getAverageLine);

	c.def("getLeftLine", &TubeOfResponse::getLeftLine);
	c.def("getRightLine", &TubeOfResponse::getRightLine);
	c.def("getFrontLine", &TubeOfResponse::getFrontLine);
	c.def("getBackLine", &TubeOfResponse::getBackLine);
	c.def("getAvgLine", &TubeOfResponse::getAvgLine);
	c.def_readwrite("thickness_z", &TubeOfResponse::thickness_z);
	c.def_readwrite("thickness_trans", &TubeOfResponse::thickness_trans);
	c.def_readwrite("isMoreHorizontalThanVertical",
	                &TubeOfResponse::isMoreHorizontalThanVertical);

	c.def("__repr__",
	      [](const TubeOfResponse& self)
	      {
		      std::stringstream ss;
		      ss << self;
		      return ss.str();
	      });
}

#endif


const StraightLineParam& TubeOfResponse::getAverageLine() const
{
	return avgLine;
}

const StraightLineParam& TubeOfResponse::getAvgLine() const
{
	return getAverageLine();
}

void TubeOfResponse::updateAvgLine()
{
	const auto average_crystal1 =
		Vector3D(
			(leftLine.point1.x + rightLine.point1.x + frontLine.point1.x +
			 backLine.point1.x) /
			4.0,
			(leftLine.point1.y + rightLine.point1.y + frontLine.point1.y +
			 backLine.point1.y) /
			4.0,
			(leftLine.point1.z + rightLine.point1.z + frontLine.point1.z +
			 backLine.point1.z) /
			4.0);

	const auto average_crystal2 =
		Vector3D(
			(leftLine.point2.x + rightLine.point2.x + frontLine.point2.x +
			 backLine.point2.x) /
			4.0,
			(leftLine.point2.y + rightLine.point2.y + frontLine.point2.y +
			 backLine.point2.y) /
			4.0,
			(leftLine.point2.z + rightLine.point2.z + frontLine.point2.z +
			 backLine.point2.z) /
			4.0);

	avgLine = StraightLineParam(average_crystal1, average_crystal2);
}

std::ostream& operator<<(std::ostream& oss, const TubeOfResponse& self)
{
	oss << "Left line: " << self.getLeftLine()
		<< "\nRight line: " << self.getRightLine()
		<< "\nFront line: " << self.getFrontLine()
		<< "\nBack line: " << self.getBackLine()
		<< "\nAverage line: " << self.getAverageLine();
	return oss;
}

TubeOfResponse::TubeOfResponse(const Vector3D& p1, const Vector3D& p2,
                               const Vector3D& n1, const Vector3D& n2,
                               float p_thickness_trans, float p_thickness_z)
{
	thickness_z = p_thickness_z;
	thickness_trans = p_thickness_trans;
	avgLine = StraightLineParam(p1, p2);
	m_n1 = n1;
	m_n2 = n2;

	Vector3D zVect(0.0, 0.0, 1.0);
	Vector3D sidesVect1 = n1.crossProduct(zVect);
	Vector3D sidesVect2 = n2.crossProduct(zVect);

	Vector3D crystal1Left = avgLine.point1 + sidesVect1 * thickness_trans /
	                            2;
	Vector3D crystal1Right =
		avgLine.point1 - sidesVect1 * thickness_trans / 2;
	Vector3D crystal1Front = avgLine.point1 - zVect * thickness_z / 2;
	Vector3D crystal1Back = avgLine.point1 + zVect * thickness_z / 2;

	sidesVect2 = sidesVect2 * (-1.0);

	Vector3D crystal2Left = avgLine.point2 + sidesVect2 * thickness_trans /
	                            2;
	Vector3D crystal2Right =
		avgLine.point2 - sidesVect2 * thickness_trans / 2;
	Vector3D crystal2Front = avgLine.point2 - zVect * thickness_z / 2;
	Vector3D crystal2Back = avgLine.point2 + zVect * thickness_z / 2;

	leftLine = StraightLineParam(crystal1Left, crystal2Left);
	rightLine = StraightLineParam(crystal1Right, crystal2Right);
	frontLine = StraightLineParam(crystal1Front, crystal2Front);
	backLine = StraightLineParam(crystal1Back, crystal2Back);

	Vector3D tubevector = avgLine.point2 - avgLine.point1;
	isMoreHorizontalThanVertical =
		(std::abs(tubevector.y) < std::abs(tubevector.x));
}

bool TubeOfResponse::clip(const Cylinder& cyl)
{
	bool allIntersect = true;
	allIntersect &= cyl.clip_line(&leftLine);
	allIntersect &= cyl.clip_line(&rightLine);
	allIntersect &= cyl.clip_line(&backLine);
	allIntersect &= cyl.clip_line(&frontLine);
	updateAvgLine();
	return allIntersect;
}