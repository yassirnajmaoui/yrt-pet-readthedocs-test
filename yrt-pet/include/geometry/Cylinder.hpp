/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/StraightLineParam.hpp"
#include "geometry/Vector3D.hpp"

class Cylinder
{

public:
	Vector3D center;
	double length_z;
	double radius;

public:
	Cylinder();
	Cylinder(const Vector3D& cent, double lz, double r);
	bool does_line_inter_cyl_infinite(const StraightLineParam* l,
	                                  Vector3D* p1, Vector3D* p2) const;
	bool does_line_inter_cyl(const StraightLineParam* l, Vector3D* p1,
	                         Vector3D* p2) const;
	bool does_line_inter_cyl(const StraightLineParam& l, Vector3D* p1,
	                         Vector3D* p2) const;
	bool clip_line(StraightLineParam* l) const;
	bool clip_line_infinite(StraightLineParam* l) const;
};
