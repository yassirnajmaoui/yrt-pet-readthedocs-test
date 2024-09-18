/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/Vector3D.hpp"

#include <vector>

class StraightLineParam
{

public:
	Vector3D point1;
	Vector3D point2;
	Vector3D current_point;
	double a, b, c, d, e, f;
	double tcur;
	/*
	  x = a*t + b
	  y = c*t + d
	  z = e*t + f
	  t in [0;1]
	*/

public:
	StraightLineParam() : a(0), b(0), c(0), d(0), e(0), f(0), tcur(0) {}
	StraightLineParam(const Vector3D& pt1, const Vector3D& pt2);
	StraightLineParam(const Vector3DFloat& pt1, const Vector3DFloat& pt2);
	virtual ~StraightLineParam() {}

	void update(const Vector3D& pt1, const Vector3D& pt2);
	void update_eq();
	bool updateCurrentPoint(double distance);
	bool isEqual(StraightLineParam& line) const;
	bool isParallel(StraightLineParam& line) const;
	float getNorm() const;
	std::vector<Vector3D> getLineParamPoints() const;
	friend std::ostream& operator<<(std::ostream& oss,
	                                const StraightLineParam& v);
};
