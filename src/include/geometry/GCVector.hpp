/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <ostream>

template<typename TFloat>
class GCVectorBase
{
public:
	GCVectorBase(TFloat xi, TFloat yi, TFloat zi);
	GCVectorBase(const GCVectorBase& v);
	GCVectorBase();
	virtual ~GCVectorBase();

	TFloat getNorm() const;
	void update(TFloat xi, TFloat yi, TFloat zi);
	GCVectorBase& operator=(const GCVectorBase& v);
	void update(const GCVectorBase& v);
	GCVectorBase& normalize();
	GCVectorBase getNormalized();
	GCVectorBase operator-(const GCVectorBase& v) const;
	GCVectorBase operator+(const GCVectorBase& vector) const;
	TFloat scalProd(const GCVectorBase& vector) const;
	GCVectorBase crossProduct(const GCVectorBase& B) const;
	void linear_transformation(const GCVectorBase& i, const GCVectorBase& j,
	                           const GCVectorBase& k);
	int argmax();
	GCVectorBase operator*(const GCVectorBase& vector) const;
	GCVectorBase operator+(TFloat scal) const;
	GCVectorBase operator-(TFloat scal) const;
	GCVectorBase operator*(TFloat scal) const;
	GCVectorBase operator/(TFloat scal) const;
	TFloat operator[](int idx) const;
	TFloat operator[](int idx);
	bool operator==(const GCVectorBase& vector) const;

public:
	TFloat x;
	TFloat y;
	TFloat z;
	bool isNormalized;
};

template <typename TFloat>
std::ostream& operator<<(std::ostream& oss, const GCVectorBase<TFloat>& v);

using GCVector = GCVectorBase<double>;

// Created to avoid type-castings
using GCVectorFloat = GCVectorBase<float>;
