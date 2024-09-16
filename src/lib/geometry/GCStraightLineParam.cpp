/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/GCStraightLineParam.hpp"

#include "geometry/GCConstants.hpp"

#include <cmath>
#include <iostream>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <sstream>
namespace py = pybind11;

void py_setup_gcstraightlineparam(py::module& m)
{
	auto c = py::class_<GCStraightLineParam>(m, "GCStraightLineParam");
	c.def(py::init<>());
	c.def(py::init<GCVector, GCVector>());
	c.def("update", &GCStraightLineParam::update);
	c.def("update_eq", &GCStraightLineParam::update_eq);
	c.def("updateCurrentPoint", &GCStraightLineParam::updateCurrentPoint);
	c.def("isEqual", &GCStraightLineParam::isEqual);
	c.def("isParallel", &GCStraightLineParam::isParallel);
	c.def("getLineParamPoints", &GCStraightLineParam::getLineParamPoints);
	c.def_readwrite("point1", &GCStraightLineParam::point1);
	c.def_readwrite("point2", &GCStraightLineParam::point2);
	c.def_readwrite("current_point", &GCStraightLineParam::current_point);
	c.def_readwrite("a", &GCStraightLineParam::a);
	c.def_readwrite("b", &GCStraightLineParam::b);
	c.def_readwrite("c", &GCStraightLineParam::c);
	c.def_readwrite("d", &GCStraightLineParam::d);
	c.def_readwrite("e", &GCStraightLineParam::e);
	c.def_readwrite("f", &GCStraightLineParam::f);
	c.def_readwrite("tcur", &GCStraightLineParam::tcur);
	c.def("__repr__",
	      [](const GCStraightLineParam& self)
	      {
		      std::stringstream ss;
		      ss << self;
		      return ss.str();
	      });
}

#endif

GCStraightLineParam::GCStraightLineParam(const GCVector& pt1,
                                         const GCVector& pt2)
{
	update(pt1, pt2);
}


void GCStraightLineParam::update(const GCVector& pt1, const GCVector& pt2)
{
	point1.update(pt1);
	point2.update(pt2);
	current_point.update(pt1);
	tcur = 0;
	update_eq();
}


void GCStraightLineParam::update_eq()
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
bool GCStraightLineParam::updateCurrentPoint(double distance)
{
	double denom = a * a + c * c + e * e;
	if (denom <= 0)
	{
		std::cerr << "Error: no solutions in "
		          << "GCStraightLineParam::updateCurrentPoint()." << std::endl;
		return false;
	}
	double delta = distance / denom;
	tcur = tcur + delta;
	current_point.x = a * tcur + b;
	current_point.y = c * tcur + d;
	current_point.z = e * tcur + f;
	return tcur >= 0 && tcur <= 1;
}

bool GCStraightLineParam::isEqual(GCStraightLineParam& line) const
{
	return fabs(a - line.a) < 10e-8 && fabs(b - line.b) < 10e-8 &&
	       fabs(c - line.c) < 10e-8 && fabs(d - line.d) < 10e-8 &&
	       fabs(e - line.e) < 10e-8 && fabs(f - line.f) < 10e-8;
}

bool GCStraightLineParam::isParallel(GCStraightLineParam& line) const
{
	/* two vectors a,b are // if: cross_product(a,b)=0
	 */
	GCVector tmp1(a, c, e);  // vector for current line
	tmp1.normalize();
	GCVector tmp2(line.a, line.c, line.e);  // vector for input line
	tmp2.normalize();
	GCVector crossProd(tmp1.y * tmp2.z - tmp1.z * tmp2.y,
	                   tmp1.z * tmp2.x - tmp1.x * tmp2.z,
	                   tmp1.x * tmp2.y - tmp1.y * tmp2.x);
	double norm = crossProd.getNorm();
	//    printf("norm %f \n",norm);
	return norm <= DOUBLE_PRECISION;
}

std::vector<GCVector> GCStraightLineParam::getLineParamPoints() const
{
	std::vector<GCVector> linePoints = {point1, point2};
	return linePoints;
}

float GCStraightLineParam::getNorm() const
{
	return (point1 - point2).getNorm();
}

std::ostream& operator<<(std::ostream& oss, const GCStraightLineParam& v)
{
	oss << "[p1: " << v.point1 << ", p2: " << v.point2 << "]";
	return oss;
}
