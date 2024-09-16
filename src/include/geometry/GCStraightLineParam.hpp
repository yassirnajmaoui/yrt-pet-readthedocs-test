/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/GCVector.hpp"

#include <vector>

class GCStraightLineParam
{

public:
	GCVector point1;
	GCVector point2;
	GCVector current_point;
	double a, b, c, d, e, f;
	double tcur;
	/*
	  x = a*t + b
	  y = c*t + d
	  z = e*t + f
	  t in [0;1]
	*/

public:
	GCStraightLineParam() : a(0), b(0), c(0), d(0), e(0), f(0), tcur(0) {}
	GCStraightLineParam(const GCVector& pt1, const GCVector& pt2);
	virtual ~GCStraightLineParam() {}

	void update(const GCVector& pt1, const GCVector& pt2);
	void update_eq();
	bool updateCurrentPoint(double distance);
	bool isEqual(GCStraightLineParam& line) const;
	bool isParallel(GCStraightLineParam& line) const;
	float getNorm() const;
	std::vector<GCVector> getLineParamPoints() const;
	friend std::ostream& operator<<(std::ostream& oss,
	                                const GCStraightLineParam& v);
};
