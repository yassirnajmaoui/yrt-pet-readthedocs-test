/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/GCStraightLineParam.hpp"
#include "geometry/GCVector.hpp"

class GCPlane
{

public:
	GCVector point1;
	GCVector point2;
	GCVector point3;
	GCVector dir;
	double a;
	double b;
	double c;
	double d;
	// equation of the plane: a.x + b.y + c.z + d = 0
public:
	GCPlane();
	GCPlane(const GCVector& pt1, const GCVector& pt2, const GCVector& pt3);
	void update(const GCVector& pt1, const GCVector& pt2, const GCVector& pt3);
	void update_eq(const GCVector& pt1, const GCVector& pt2, const GCVector& pt3);
	bool isCoplanar(const GCVector& pt) const;
	bool isParrallel(const GCStraightLineParam& line) const;
	GCVector findInterLine(const GCStraightLineParam& line) const;
};
