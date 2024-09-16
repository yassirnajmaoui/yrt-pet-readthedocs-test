/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/GCVector.hpp"
#include <ostream>

class GCMatrix
{
public:
	GCMatrix(double a00, double a01, double a02, double a10, double a11,
	         double a12, double a20, double a21, double a22);
	GCMatrix(const GCMatrix& v);
	GCMatrix();
	static GCMatrix identity();

	void update(double a00, double a01, double a02, double a10, double a11,
	            double a12, double a20, double a21, double a22);
	GCMatrix operator=(GCMatrix v);
	void update(const GCMatrix& v);
	GCMatrix operator-(GCMatrix v);
	GCMatrix operator+(GCMatrix matrix);
	GCMatrix operator*(GCMatrix matrix);
	GCVector operator*(GCVector vector);
	GCMatrix operator+(double scal);
	GCMatrix operator-(double scal);
	GCMatrix operator*(double scal);
	GCMatrix operator/(double scal);
	bool operator==(GCMatrix matrix);
	friend std::ostream& operator<<(std::ostream& oss, const GCMatrix& v);

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
