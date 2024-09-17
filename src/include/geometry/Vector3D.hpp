/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <ostream>

template<typename TFloat>
class Vector3DBase
{
public:
	Vector3DBase(TFloat xi, TFloat yi, TFloat zi);
	Vector3DBase(const Vector3DBase& v);
	Vector3DBase();
	virtual ~Vector3DBase();

	TFloat getNorm() const;
	void update(TFloat xi, TFloat yi, TFloat zi);
	Vector3DBase& operator=(const Vector3DBase& v);
	void update(const Vector3DBase& v);
	Vector3DBase& normalize();
	Vector3DBase getNormalized();
	Vector3DBase operator-(const Vector3DBase& v) const;
	Vector3DBase operator+(const Vector3DBase& vector) const;
	TFloat scalProd(const Vector3DBase& vector) const;
	Vector3DBase crossProduct(const Vector3DBase& B) const;
	void linear_transformation(const Vector3DBase& i, const Vector3DBase& j,
	                           const Vector3DBase& k);
	int argmax();
	Vector3DBase operator*(const Vector3DBase& vector) const;
	Vector3DBase operator+(TFloat scal) const;
	Vector3DBase operator-(TFloat scal) const;
	Vector3DBase operator*(TFloat scal) const;
	Vector3DBase operator/(TFloat scal) const;
	TFloat operator[](int idx) const;
	TFloat operator[](int idx);
	bool operator==(const Vector3DBase& vector) const;

	template<typename TargetType>
	Vector3DBase<TargetType> to() const;

public:
	TFloat x;
	TFloat y;
	TFloat z;
	bool isNormalized;
};

template <typename TFloat>
std::ostream& operator<<(std::ostream& oss, const Vector3DBase<TFloat>& v);

using Vector3D = Vector3DBase<double>;

// Created to avoid type-castings
using Vector3DFloat = Vector3DBase<float>;
