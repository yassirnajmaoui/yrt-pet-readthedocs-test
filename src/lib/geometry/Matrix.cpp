/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/Matrix.hpp"

#include "geometry/Constants.hpp"

#include <cmath>
#include <iostream>


Matrix::Matrix(double a00, double a01, double a02, double a10, double a11,
               double a12, double a20, double a21, double a22)
	: m_a00(a00),
	  m_a01(a01),
	  m_a02(a02),
	  m_a10(a10),
	  m_a11(a11),
	  m_a12(a12),
	  m_a20(a20),
	  m_a21(a21),
	  m_a22(a22) {}


Matrix::Matrix(const Matrix& v)
{
	m_a00 = v.m_a00;
	m_a01 = v.m_a01;
	m_a02 = v.m_a02;
	m_a10 = v.m_a10;
	m_a11 = v.m_a11;
	m_a12 = v.m_a12;
	m_a20 = v.m_a20;
	m_a21 = v.m_a21;
	m_a22 = v.m_a22;
}


Matrix::Matrix()
	: m_a00(0.0),
	  m_a01(0.0),
	  m_a02(0.0),
	  m_a10(0.0),
	  m_a11(0.0),
	  m_a12(0.0),
	  m_a20(0.0),
	  m_a21(0.0),
	  m_a22(0.0) {}

Matrix Matrix::identity()
{
	return Matrix{1, 0, 0, 0, 1, 0, 0, 0, 1};
}


void Matrix::update(double a00, double a01, double a02, double a10,
                    double a11, double a12, double a20, double a21,
                    double a22)
{
	m_a00 = a00;
	m_a01 = a01;
	m_a02 = a02;
	m_a10 = a10;
	m_a11 = a11;
	m_a12 = a12;
	m_a20 = a20;
	m_a21 = a21;
	m_a22 = a22;
}

Matrix Matrix::operator=(Matrix v)
{
	Matrix vv;

	vv.m_a00 = v.m_a00;
	vv.m_a01 = v.m_a01;
	vv.m_a02 = v.m_a02;
	vv.m_a10 = v.m_a10;
	vv.m_a11 = v.m_a11;
	vv.m_a12 = v.m_a12;
	vv.m_a20 = v.m_a20;
	vv.m_a21 = v.m_a21;
	vv.m_a22 = v.m_a22;

	return vv;
}


// update 3:
void Matrix::update(const Matrix& v)
{
	m_a00 = v.m_a00;
	m_a01 = v.m_a01;
	m_a02 = v.m_a02;
	m_a10 = v.m_a10;
	m_a11 = v.m_a11;
	m_a12 = v.m_a12;
	m_a20 = v.m_a20;
	m_a21 = v.m_a21;
	m_a22 = v.m_a22;
}

Matrix Matrix::operator-(Matrix v)
{

	Matrix res;

	res.m_a00 = m_a00 - v.m_a00;
	res.m_a01 = m_a01 - v.m_a01;
	res.m_a02 = m_a02 - v.m_a02;
	res.m_a10 = m_a10 - v.m_a10;
	res.m_a11 = m_a11 - v.m_a11;
	res.m_a12 = m_a12 - v.m_a12;
	res.m_a20 = m_a20 - v.m_a20;
	res.m_a21 = m_a21 - v.m_a21;
	res.m_a22 = m_a22 - v.m_a22;

	return res;
}


Matrix Matrix::operator+(Matrix v)
{

	Matrix res;

	res.m_a00 = m_a00 + v.m_a00;
	res.m_a01 = m_a01 + v.m_a01;
	res.m_a02 = m_a02 + v.m_a02;
	res.m_a10 = m_a10 + v.m_a10;
	res.m_a11 = m_a11 + v.m_a11;
	res.m_a12 = m_a12 + v.m_a12;
	res.m_a20 = m_a20 + v.m_a20;
	res.m_a21 = m_a21 + v.m_a21;
	res.m_a22 = m_a22 + v.m_a22;

	return res;
}

Matrix Matrix::operator*(Matrix matrix)
{
	Matrix res;

	res.m_a00 = m_a00 * matrix.m_a00;
	res.m_a01 = m_a01 * matrix.m_a01;
	res.m_a02 = m_a02 * matrix.m_a02;
	res.m_a10 = m_a10 * matrix.m_a10;
	res.m_a11 = m_a11 * matrix.m_a11;
	res.m_a12 = m_a12 * matrix.m_a12;
	res.m_a20 = m_a20 * matrix.m_a20;
	res.m_a21 = m_a21 * matrix.m_a21;
	res.m_a22 = m_a22 * matrix.m_a22;

	return res;
}

Vector3D Matrix::operator*(const Vector3D& vector)
{
	Vector3D res;

	res.x = m_a00 * vector.x + m_a01 * vector.y + m_a02 * vector.z;
	res.y = m_a10 * vector.x + m_a11 * vector.y + m_a12 * vector.z;
	res.z = m_a20 * vector.x + m_a21 * vector.y + m_a22 * vector.z;

	return res;
}


Matrix Matrix::operator+(double scal)
{
	Matrix res;

	res.m_a00 = m_a00 + scal;
	res.m_a01 = m_a01 + scal;
	res.m_a02 = m_a02 + scal;
	res.m_a10 = m_a10 + scal;
	res.m_a11 = m_a11 + scal;
	res.m_a12 = m_a12 + scal;
	res.m_a20 = m_a20 + scal;
	res.m_a21 = m_a21 + scal;
	res.m_a22 = m_a22 + scal;

	return res;
}


Matrix Matrix::operator-(double scal)
{
	Matrix res;

	res.m_a00 = m_a00 - scal;
	res.m_a01 = m_a01 - scal;
	res.m_a02 = m_a02 - scal;
	res.m_a10 = m_a10 - scal;
	res.m_a11 = m_a11 - scal;
	res.m_a12 = m_a12 - scal;
	res.m_a20 = m_a20 - scal;
	res.m_a21 = m_a21 - scal;
	res.m_a22 = m_a22 - scal;

	return res;
}


Matrix Matrix::operator*(double scal)
{
	Matrix res;

	res.m_a00 = m_a00 * scal;
	res.m_a01 = m_a01 * scal;
	res.m_a02 = m_a02 * scal;
	res.m_a10 = m_a10 * scal;
	res.m_a11 = m_a11 * scal;
	res.m_a12 = m_a12 * scal;
	res.m_a20 = m_a20 * scal;
	res.m_a21 = m_a21 * scal;
	res.m_a22 = m_a22 * scal;

	return res;
}


Matrix Matrix::operator/(double scal)
{
	Matrix res;

	if (fabs(scal) > DOUBLE_PRECISION)
	{
		res.m_a00 = m_a00 / scal;
		res.m_a01 = m_a01 / scal;
		res.m_a02 = m_a02 / scal;
		res.m_a10 = m_a10 / scal;
		res.m_a11 = m_a11 / scal;
		res.m_a12 = m_a12 / scal;
		res.m_a20 = m_a20 / scal;
		res.m_a21 = m_a21 / scal;
		res.m_a22 = m_a22 / scal;
	}
	else
	{
		res.m_a00 = LARGE_VALUE;
		res.m_a01 = LARGE_VALUE;
		res.m_a02 = LARGE_VALUE;
		res.m_a10 = LARGE_VALUE;
		res.m_a11 = LARGE_VALUE;
		res.m_a12 = LARGE_VALUE;
		res.m_a20 = LARGE_VALUE;
		res.m_a21 = LARGE_VALUE;
		res.m_a22 = LARGE_VALUE;
	}

	return res;
}


// return true if matrices are the same:
bool Matrix::operator==(Matrix matrix)
{
	double sqr_norm = (m_a00 - matrix.m_a00) * (m_a00 - matrix.m_a00) +
	                  (m_a01 - matrix.m_a01) * (m_a01 - matrix.m_a01) +
	                  (m_a02 - matrix.m_a02) * (m_a02 - matrix.m_a02) +
	                  (m_a10 - matrix.m_a10) * (m_a10 - matrix.m_a10) +
	                  (m_a11 - matrix.m_a11) * (m_a11 - matrix.m_a11) +
	                  (m_a12 - matrix.m_a12) * (m_a12 - matrix.m_a12) +
	                  (m_a20 - matrix.m_a20) * (m_a20 - matrix.m_a20) +
	                  (m_a21 - matrix.m_a21) * (m_a21 - matrix.m_a21) +
	                  (m_a22 - matrix.m_a22) * (m_a22 - matrix.m_a22);


	return sqrt(sqr_norm) < DOUBLE_PRECISION;
}

std::ostream& operator<<(std::ostream& oss, const Matrix& v)
{
	oss << "("
		<< "(" << v.m_a00 << ", " << v.m_a01 << ", " << v.m_a02 << "), ("
		<< v.m_a10 << ", " << v.m_a11 << ", " << v.m_a12 << "), (" << v.m_a20
		<< ", " << v.m_a21 << ", " << v.m_a22 << ")";
	return oss;
}