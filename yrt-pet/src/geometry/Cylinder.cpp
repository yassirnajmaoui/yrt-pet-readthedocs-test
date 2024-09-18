/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/Cylinder.hpp"

#include "geometry/Constants.hpp"
#include "geometry/Vector3D.hpp"

#include <cmath>

// constructors:
Cylinder::Cylinder(const Vector3D& cent, double lz, double r)
{
	center = cent;
	length_z = lz;
	radius = r;
}

Cylinder::Cylinder()
{
	center = Vector3D{};
	length_z = 0.0;
	radius = 0.0;
}

bool Cylinder::does_line_inter_cyl(const StraightLineParam& l, Vector3D* p1,
                                     Vector3D* p2) const
{
	return does_line_inter_cyl(&l, p1, p2);
}

bool Cylinder::does_line_inter_cyl_infinite(const StraightLineParam* l,
                                              Vector3D* p1, Vector3D* p2) const
{
	double a = l->a, b = l->b, c = l->c, d = l->d, e = l->e, f = l->f;
	double A = a * a + c * c;
	double B = a * (b - center.x) + c * (d - center.y);
	B *= 2;
	double C = GET_SQ(b - center.x) + GET_SQ(d - center.y) - GET_SQ(radius);
	double delta = B * B - 4 * A * C;
	if (delta < 0)
		return false;
	double t1 = -B - sqrt(delta);
	t1 /= 2 * A;
	double t2 = -B + sqrt(delta);
	t2 /= 2 * A;
	p1->x = a * t1 + b;
	p1->y = c * t1 + d;
	p1->z = e * t1 + f;

	p2->x = a * t2 + b;
	p2->y = c * t2 + d;
	p2->z = e * t2 + f;
	return true;
}

bool Cylinder::does_line_inter_cyl(const StraightLineParam* l, Vector3D* p1,
                                     Vector3D* p2) const
{
	bool infinite_ret = does_line_inter_cyl_infinite(l, p1, p2);
	if (!infinite_ret)
		return false;
	if (fabs(p1->z - center.z) > length_z / 2 ||
	    fabs(p2->z - center.z) > length_z / 2)
		return false;
	else
		return true;
}

bool Cylinder::clip_line(StraightLineParam* l) const
{
	Vector3D p1, p2;
	if (!does_line_inter_cyl(l, &p1, &p2))
		return false;
	l->update(p1, p2);
	return true;
}

bool Cylinder::clip_line_infinite(StraightLineParam* l) const
{
	Vector3D p1, p2;
	if (!does_line_inter_cyl_infinite(l, &p1, &p2))
		return false;
	l->update(p1, p2);
	return true;
}
