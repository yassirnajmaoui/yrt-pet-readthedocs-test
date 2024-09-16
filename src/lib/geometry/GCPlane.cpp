/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/GCPlane.hpp"

#include "geometry/GCConstants.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdio.h>

GCPlane::GCPlane() {}


GCPlane::GCPlane(const GCVector& pt1, const GCVector& pt2, const GCVector& pt3)
    : point1(pt1), point2(pt2), point3(pt3)
{
	GCVector vector_pos1, vector_pos2;
	// check that these points define a plane:
	vector_pos1.update(pt3.x - pt1.x, pt3.y - pt1.y, pt3.z - pt1.z);
	vector_pos2.update(pt3.x - pt2.x, pt3.y - pt2.y, pt3.z - pt2.z);
	dir = vector_pos1 * vector_pos2;
	if (dir.getNorm() < DOUBLE_PRECISION)
	{
		std::cout << "\nThe 3 points in input of GCPlane::GCPlane() do not "
		             "define a plane.\n";
		std::cout << "point1 = ( " << pt1.x << ", " << pt1.y << ", " << pt1.z
		          << " )  point2 = ( " << pt2.x << ", " << pt2.y << ", "
		          << pt2.z << " )  point3 = ( " << pt3.x << ", " << pt3.y
		          << ", " << pt3.z << " )\n\n";
		exit(-1);
	}
	// find equation of the plane:
	update_eq(pt1, pt2, pt3);
}


// update function:
void GCPlane::update(const GCVector& pt1, const GCVector& pt2,
                     const GCVector& pt3)
{
	GCVector vector_pos1, vector_pos2;
	point1.update(pt1.x, pt1.y, pt1.z);
	point2.update(pt2.x, pt2.y, pt2.z);
	point3.update(pt3.x, pt3.y, pt3.z);
	// check that these points define a plane:
	vector_pos1.update(pt3.x - pt1.x, pt3.y - pt1.y, pt3.z - pt1.z);
	vector_pos2.update(pt3.x - pt2.x, pt3.y - pt2.y, pt3.z - pt2.z);
	const GCVector dir_tmp = vector_pos1 * vector_pos2;
	dir.update(dir_tmp.x, dir_tmp.y, dir_tmp.z);
	if (dir.getNorm() < DOUBLE_PRECISION)
	{
		std::cout << "\nThe 3 points in input of GCPlane::GCPlane() do not "
		             "define a plane.\n";
		std::cout << "point1 = ( " << pt1.x << ", " << pt1.y << ", " << pt1.z
		          << " )  point2 = ( " << pt2.x << ", " << pt2.y << ", "
		          << pt2.z << " )  point3 = ( " << pt3.x << ", " << pt3.y
		          << ", " << pt3.z << " )\n\n",
		    getchar();
		exit(-1);
	}
	// update equation:
	update_eq(pt1, pt2, pt3);
}

// update equation of the plane using these 3 points:
void GCPlane::update_eq(const GCVector& pt1, const GCVector& pt2,
                        const GCVector& pt3)
{
	// find equation of the plane
	const double x1 = pt1.x;
	const double x2 = pt2.x;
	const double x3 = pt3.x;
	const double y1 = pt1.y;
	const double y2 = pt2.y;
	const double y3 = pt3.y;
	const double z1 = pt1.z;
	const double z2 = pt2.z;
	const double z3 = pt3.z;
	a = y1 * (z2 - z3) + y2 * (z3 - z1) + y3 * (z1 - z2);
	b = z1 * (x2 - x3) + z2 * (x3 - x1) + z3 * (x1 - x2);
	c = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2);
	d = -x1 * a - y1 * b - z1 * c;
}


// returns true if "point" is in the plane
bool GCPlane::isCoplanar(const GCVector& pt) const
{
	bool isCop = false;
	const GCVector vector_pos = point3 - pt;
	const double scalProd = dir.scalProd(vector_pos);
	if (fabs(scalProd) < DOUBLE_PRECISION)
	{
		isCop = true;
	}
	return isCop;
}


// returns true if "line" is parrallel to the current plan
bool GCPlane::isParrallel(const GCStraightLineParam& line) const
{
	const double test = a * line.a + b * line.c + c * line.e;
	if (fabs(test) < 10e-8)
		return true;
	else
		return false;
}


// this function returns the intersection point of the current plane
// with a line defined by two points "point1" and "point2".
// The coords. of the inter point are equal to LARGE_VALUE+1
// if the line is parrallel to the plane.
GCVector GCPlane::findInterLine(const GCStraightLineParam& line) const
{
	const double denom = a * line.a + b * line.c + c * line.e;
	double t = a * line.b + b * line.d + c * line.f + d;
	GCVector tmp;
	if (fabs(denom) < 10e-8)
	{
		tmp.update(LARGE_VALUE + 1, LARGE_VALUE + 1, LARGE_VALUE + 1);
	}
	else
	{
		t /= -denom;
		tmp.x = line.a * t + line.b;
		tmp.y = line.c * t + line.d;
		tmp.z = line.e * t + line.f;
	}
	return tmp;
}
