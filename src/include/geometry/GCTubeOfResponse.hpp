/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/GCStraightLineParam.hpp"

class GCCylinder;

class GCTubeOfResponse
{
public:
	GCTubeOfResponse(const GCVector& p1, const GCVector& p2, const GCVector& n1,
	                 const GCVector& n2, float p_thickness_trans,
	                 float p_thickness_z);

	const GCStraightLineParam& getLeftLine() const { return leftLine; }
	const GCStraightLineParam& getRightLine() const { return rightLine; }
	const GCStraightLineParam& getFrontLine() const { return frontLine; }
	const GCStraightLineParam& getBackLine() const { return backLine; }
	void setLeftLine(GCStraightLineParam l)
	{
		leftLine = l;
		updateAvgLine();
	}
	void setRightLine(GCStraightLineParam l)
	{
		rightLine = l;
		updateAvgLine();
	}
	void setFrontLine(GCStraightLineParam l)
	{
		frontLine = l;
		updateAvgLine();
	}
	void setBackLine(GCStraightLineParam l)
	{
		backLine = l;
		updateAvgLine();
	}
	const GCStraightLineParam& getAverageLine() const;
	const GCStraightLineParam& getAvgLine() const;  // Alias
	bool clip(const GCCylinder& cyl);
	friend std::ostream& operator<<(std::ostream& oss,
	                                const GCTubeOfResponse& v);

private:
	void updateAvgLine();

private:
	GCStraightLineParam leftLine;
	GCStraightLineParam rightLine;
	GCStraightLineParam frontLine;
	GCStraightLineParam backLine;
	GCStraightLineParam avgLine;
	// The placement of lines is described in the ASV paper
	// Here's a scheme: https://imgur.com/a/VIvuEvq
	// ("rear" has been replaced by "back")
public:
	GCVector m_n1;
	GCVector m_n2;
	float thickness_z, thickness_trans;
	bool isMoreHorizontalThanVertical;
};
