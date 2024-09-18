/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/Vector3D.hpp"
#include <ostream>

class Matrix
{
public:
    Matrix(double a00, double a01, double a02, double a10, double a11,
           double a12, double a20, double a21, double a22);
    Matrix(const Matrix& v);
    Matrix();
    static Matrix identity();

    void update(double a00, double a01, double a02, double a10, double a11,
                double a12, double a20, double a21, double a22);
    Matrix operator=(Matrix v);
    void update(const Matrix& v);
    Matrix operator-(Matrix v);
    Matrix operator+(Matrix matrix);
    Matrix operator*(Matrix matrix);
    Vector3D operator*(const Vector3D& vector);
    Matrix operator+(double scal);
    Matrix operator-(double scal);
    Matrix operator*(double scal);
    Matrix operator/(double scal);
    bool operator==(Matrix matrix);
    friend std::ostream& operator<<(std::ostream& oss, const Matrix& v);

private:
    double m_a00;
    double m_a01;
    double m_a02;
    double m_a10;
    double m_a11;
    double m_a12;
    double m_a20;
    double m_a21;
    double m_a22;
};
