/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/GCStraightLineParam.hpp"
#include "geometry/GCVector.hpp"

class GCCylinder
{

public:
	GCVector center;
	double length_z;
	double radius;

public:
	GCCylinder();
	GCCylinder(GCVector cent, double lz, double r);
	bool does_line_inter_cyl_infinite(const GCStraightLineParam* l,
	                                  GCVector* p1, GCVector* p2) const;
	bool does_line_inter_cyl(const GCStraightLineParam* l, GCVector* p1,
	                         GCVector* p2) const;
	bool does_line_inter_cyl(const GCStraightLineParam& l, GCVector* p1,
	                         GCVector* p2) const;
	bool clip_line(GCStraightLineParam* l) const;
	bool clip_line_infinite(GCStraightLineParam* l) const;
};
