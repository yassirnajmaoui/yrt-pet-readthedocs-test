/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/StraightLineParam.hpp"

class Cylinder;

class TubeOfResponse
{
public:
    TubeOfResponse(const Vector3D& p1, const Vector3D& p2, const Vector3D& n1,
                   const Vector3D& n2, float p_thickness_trans,
                   float p_thickness_z);

    const StraightLineParam& getLeftLine() const { return leftLine; }
    const StraightLineParam& getRightLine() const { return rightLine; }
    const StraightLineParam& getFrontLine() const { return frontLine; }
    const StraightLineParam& getBackLine() const { return backLine; }

    void setLeftLine(StraightLineParam l)
    {
        leftLine = l;
        updateAvgLine();
    }

    void setRightLine(StraightLineParam l)
    {
        rightLine = l;
        updateAvgLine();
    }

    void setFrontLine(StraightLineParam l)
    {
        frontLine = l;
        updateAvgLine();
    }

    void setBackLine(StraightLineParam l)
    {
        backLine = l;
        updateAvgLine();
    }

    const StraightLineParam& getAverageLine() const;
    const StraightLineParam& getAvgLine() const; // Alias
    bool clip(const Cylinder& cyl);
    friend std::ostream& operator<<(std::ostream& oss,
                                    const TubeOfResponse& v);

private:
    void updateAvgLine();

private:
    StraightLineParam leftLine;
    StraightLineParam rightLine;
    StraightLineParam frontLine;
    StraightLineParam backLine;
    StraightLineParam avgLine;
    // The placement of lines is described in the ASV paper
    // Here's a scheme: https://imgur.com/a/VIvuEvq
    // ("rear" has been replaced by "back")
public:
    Vector3D m_n1;
    Vector3D m_n2;
    float thickness_z, thickness_trans;
    bool isMoreHorizontalThanVertical;
};
