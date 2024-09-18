/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/StraightLineParam.hpp"

#include "geometry/Constants.hpp"

#include <cmath>
#include <iostream>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <sstream>
namespace py = pybind11;

void py_setup_straightlineparam(py::module& m)
{
	auto c = py::class_<StraightLineParam>(m, "StraightLineParam");
	c.def(py::init<>());
	c.def(py::init<Vector3D, Vector3D>());
	c.def(py::init<Vector3DFloat, Vector3DFloat>());
	c.def("update", &StraightLineParam::update);
	c.def("update_eq", &StraightLineParam::update_eq);
	c.def("updateCurrentPoint", &StraightLineParam::updateCurrentPoint);
	c.def("isEqual", &StraightLineParam::isEqual);
	c.def("isParallel", &StraightLineParam::isParallel);
	c.def("getLineParamPoints", &StraightLineParam::getLineParamPoints);
	c.def_readwrite("point1", &StraightLineParam::point1);
	c.def_readwrite("point2", &StraightLineParam::point2);
	c.def_readwrite("current_point", &StraightLineParam::current_point);
	c.def_readwrite("a", &StraightLineParam::a);
	c.def_readwrite("b", &StraightLineParam::b);
	c.def_readwrite("c", &StraightLineParam::c);
	c.def_readwrite("d", &StraightLineParam::d);
	c.def_readwrite("e", &StraightLineParam::e);
	c.def_readwrite("f", &StraightLineParam::f);
	c.def_readwrite("tcur", &StraightLineParam::tcur);
	c.def("__repr__",
	      [](const StraightLineParam& self)
	      {
		      std::stringstream ss;
		      ss << self;
		      return ss.str();
	      });
}

#endif

StraightLineParam::StraightLineParam(const Vector3D& pt1,
                                     const Vector3D& pt2)
{
	update(pt1, pt2);
}

StraightLineParam::StraightLineParam(const Vector3DFloat& pt1,
                                     const Vector3DFloat& pt2)
{
	const auto pt1_double = Vector3D{pt1.x, pt1.y, pt1.z};
	const auto pt2_double = Vector3D{pt2.x, pt2.y, pt2.z};
	update(pt1_double, pt2_double);
}


void StraightLineParam::update(const Vector3D& pt1, const Vector3D& pt2)
{
	point1.update(pt1);
	point2.update(pt2);
	current_point.update(pt1);
	tcur = 0;
	update_eq();
}


void StraightLineParam::update_eq()
{
	b = point1.x;
	d = point1.y;
	f = point1.z;
	a = point2.x - b;
	c = point2.y - d;
	e = point2.z - f;
}

// update position of the current point by moving it
// of distance d along the line. Returns true if current
// point is still between point1 and point2, false otherwise.
// If distance > 0 => move in the dir point1, point2;
// if distance < 0 => move in the dir point2, point1;
bool StraightLineParam::updateCurrentPoint(double distance)
{
	double denom = a * a + c * c + e * e;
	if (denom <= 0)
	{
		std::cerr << "Error: no solutions in "
			<< "StraightLineParam::updateCurrentPoint()." << std::endl;
		return false;
	}
	const double delta = distance / denom;
	tcur = tcur + delta;
	current_point.x = a * tcur + b;
	current_point.y = c * tcur + d;
	current_point.z = e * tcur + f;
	return tcur >= 0 && tcur <= 1;
}

bool StraightLineParam::isEqual(StraightLineParam& line) const
{
	return fabs(a - line.a) < 10e-8 && fabs(b - line.b) < 10e-8 &&
	       fabs(c - line.c) < 10e-8 && fabs(d - line.d) < 10e-8 &&
	       fabs(e - line.e) < 10e-8 && fabs(f - line.f) < 10e-8;
}

bool StraightLineParam::isParallel(StraightLineParam& line) const
{
	/* two vectors a,b are // if: cross_product(a,b)=0
	 */
	Vector3D tmp1(a, c, e); // vector for current line
	tmp1.normalize();
	Vector3D tmp2(line.a, line.c, line.e); // vector for input line
	tmp2.normalize();
	Vector3D crossProd(tmp1.y * tmp2.z - tmp1.z * tmp2.y,
	                       tmp1.z * tmp2.x - tmp1.x * tmp2.z,
	                       tmp1.x * tmp2.y - tmp1.y * tmp2.x);
	double norm = crossProd.getNorm();
	//    printf("norm %f \n",norm);
	return norm <= DOUBLE_PRECISION;
}

std::vector<Vector3D> StraightLineParam::getLineParamPoints() const
{
	std::vector<Vector3D> linePoints = {point1, point2};
	return linePoints;
}

float StraightLineParam::getNorm() const
{
	return (point1 - point2).getNorm();
}

std::ostream& operator<<(std::ostream& oss, const StraightLineParam& v)
{
	oss << "[p1: " << v.point1 << ", p2: " << v.point2 << "]";
	return oss;
}