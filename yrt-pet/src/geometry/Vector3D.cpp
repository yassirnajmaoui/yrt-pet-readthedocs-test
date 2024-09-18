/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/Vector3D.hpp"

#include "geometry/Constants.hpp"

#include <cmath>

#if BUILD_PYBIND11
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <sstream>

namespace py = pybind11;

template <typename TFloat>
void py_setup_vector3dbase(py::module& m)
{
	std::string className = "Vector3D";
	if (typeid(TFloat) == typeid(float))
	{
		className += "Float";
	}

	auto c = py::class_<Vector3DBase<TFloat>>(m, className.c_str());
	c.def(py::init<>());
	c.def(py::init<TFloat, TFloat, TFloat>());
	c.def(py::init<const Vector3DBase<TFloat>&>());
	c.def("getNorm", &Vector3DBase<TFloat>::getNorm);
	c.def("update",
	      static_cast<void (Vector3DBase<TFloat>::*)(TFloat, TFloat, TFloat)>(
		      &Vector3DBase<TFloat>::update));
	c.def("update",
	      static_cast<void (Vector3DBase<TFloat>::*)(
		      const Vector3DBase<TFloat>&)>(&Vector3DBase<TFloat>::update));
	c.def("normalize", &Vector3DBase<TFloat>::normalize);
	c.def("__sub__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(
		      const Vector3DBase<TFloat>&) const>(
		      &Vector3DBase<TFloat>::operator-),
	      py::is_operator());
	c.def("__add__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(
		      const Vector3DBase<TFloat>&) const>(
		      &Vector3DBase<TFloat>::operator+),
	      py::is_operator());
	c.def("__mul__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(
		      const Vector3DBase<TFloat>&) const>(
		      &Vector3DBase<TFloat>::operator*),
	      py::is_operator());
	c.def("__sub__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(TFloat)
		      const>(&Vector3DBase<TFloat>::operator-),
	      py::is_operator());
	c.def("__add__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(TFloat)
		      const>(&Vector3DBase<TFloat>::operator+),
	      py::is_operator());
	c.def("__mul__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(TFloat)
		      const>(&Vector3DBase<TFloat>::operator*),
	      py::is_operator());
	c.def("__div__",
	      static_cast<Vector3DBase<TFloat> (Vector3DBase<TFloat>::*)(TFloat)
		      const>(&Vector3DBase<TFloat>::operator/),
	      py::is_operator());
	c.def(
		"__eq__",
		static_cast<bool (Vector3DBase<TFloat>::*)(const Vector3DBase<TFloat>&)
			const>(&Vector3DBase<TFloat>::operator==),
		py::is_operator());
	c.def("__repr__",
	      [](const Vector3DBase<TFloat>& self)
	      {
		      std::stringstream ss;
		      ss << self;
		      return ss.str();
	      });
	c.def_readwrite("x", &Vector3DBase<TFloat>::x);
	c.def_readwrite("y", &Vector3DBase<TFloat>::y);
	c.def_readwrite("z", &Vector3DBase<TFloat>::z);
	c.def_readwrite("isNormalized", &Vector3DBase<TFloat>::isNormalized);
	c.def("getNormalized", &Vector3DBase<TFloat>::getNormalized);
}

void py_setup_vector3dall(py::module& m)
{
	py_setup_vector3dbase<float>(m);
	py_setup_vector3dbase<double>(m);
}

#endif

template <typename TFloat>
Vector3DBase<TFloat>::Vector3DBase(TFloat xi, TFloat yi, TFloat zi)
	: x(xi),
	  y(yi),
	  z(zi),
	  isNormalized(false) {}


template <typename TFloat>
Vector3DBase<TFloat>::Vector3DBase(const Vector3DBase& v)
{
	x = v.x;
	y = v.y;
	z = v.z;
	isNormalized = v.isNormalized;
}


template <typename TFloat>
Vector3DBase<TFloat>::Vector3DBase()
	: x(0.),
	  y(0.),
	  z(0.),
	  isNormalized(false) {}


// destructor
template <typename TFloat>
Vector3DBase<TFloat>::~Vector3DBase() = default;

template <typename TFloat>
TFloat Vector3DBase<TFloat>::getNorm() const
{
	TFloat norm;

	norm = sqrt(x * x + y * y + z * z);

	return norm;
}


template <typename TFloat>
void Vector3DBase<TFloat>::update(TFloat xi, TFloat yi, TFloat zi)
{
	x = xi;
	y = yi;
	z = zi;
	isNormalized = false;
}


template <typename TFloat>
Vector3DBase<TFloat>& Vector3DBase<TFloat>::operator=(const Vector3DBase& v)
{
	x = v.x;
	y = v.y;
	z = v.z;
	isNormalized = v.isNormalized;

	return *this;
}


// update 3:
template <typename TFloat>
void Vector3DBase<TFloat>::update(const Vector3DBase& v)
{
	x = v.x;
	y = v.y;
	z = v.z;
	isNormalized = v.isNormalized;
}

template <typename TFloat>
Vector3DBase<TFloat>& Vector3DBase<TFloat>::normalize()
{
	TFloat norm;

	norm = getNorm();

	if (norm > DOUBLE_PRECISION)
	{
		x = x / norm;
		y = y / norm;
		z = z / norm;
	}
	else
	{
		x = 0.;
		y = 0.;
		z = 0.;
	}

	isNormalized = true;
	return *this;
}

template <typename TFloat>
Vector3DBase<TFloat> Vector3DBase<TFloat>::getNormalized()
{
	TFloat norm;
	Vector3DBase res;

	norm = getNorm();

	if (norm > DOUBLE_PRECISION)
	{
		res.x = x / norm;
		res.y = y / norm;
		res.z = z / norm;
	}
	else
	{
		res.x = 0.;
		res.y = 0.;
		res.z = 0.;
	}

	res.isNormalized = true;
	return res;
}

template <typename TFloat>
Vector3DBase<TFloat>
	Vector3DBase<TFloat>::operator-(const Vector3DBase& v) const
{
	Vector3DBase res;

	res.x = x - v.x;
	res.y = y - v.y;
	res.z = z - v.z;

	return res;
}


template <typename TFloat>
Vector3DBase<TFloat>
	Vector3DBase<TFloat>::operator+(const Vector3DBase& vector) const
{
	Vector3DBase res;

	res.x = x + vector.x;
	res.y = y + vector.y;
	res.z = z + vector.z;

	return res;
}


template <typename TFloat>
TFloat Vector3DBase<TFloat>::scalProd(const Vector3DBase& vector) const
{
	TFloat res;

	res = x * vector.x + y * vector.y + z * vector.z;

	return res;
}

template <typename TFloat>
Vector3DBase<TFloat>
	Vector3DBase<TFloat>::crossProduct(const Vector3DBase& B) const
{
	Vector3DBase P(0.0, 0.0, 0.0);
	P.x = this->y * B.z - this->z * B.y;
	P.y = this->z * B.x - this->x * B.z;
	P.z = this->x * B.y - this->y * B.x;
	return P;
}

template <typename TFloat>
void Vector3DBase<TFloat>::linear_transformation(const Vector3DBase& i,
                                                 const Vector3DBase& j,
                                                 const Vector3DBase& k)
{
	this->x = i.x * this->x + j.x * this->y + k.x * this->z;
	this->y = i.y * this->x + j.y * this->y + k.y * this->z;
	this->z = i.z * this->x + j.z * this->y + k.z * this->z;
}

template <typename TFloat>
int Vector3DBase<TFloat>::argmax()
{
	return (x > y) ? ((x > z) ? 0 : 2) : ((y > z) ? 1 : 2);
}

template <typename TFloat>
TFloat Vector3DBase<TFloat>::operator[](int idx) const
{
	if (idx == 0)
		return x;
	else if (idx == 1)
		return y;
	else if (idx == 2)
		return z;
	else
		return NAN;
}

template <typename TFloat>
TFloat Vector3DBase<TFloat>::operator[](int idx)
{
	if (idx == 0)
		return x;
	else if (idx == 1)
		return y;
	else if (idx == 2)
		return z;
	else
		return NAN;
}


template <typename TFloat>
Vector3DBase<TFloat>
	Vector3DBase<TFloat>::operator*(const Vector3DBase& vector) const
{
	Vector3DBase res;

	res.x = y * vector.z - z * vector.y;
	res.y = z * vector.x - x * vector.z;
	res.z = x * vector.y - y * vector.x;

	return res;
}


template <typename TFloat>
Vector3DBase<TFloat> Vector3DBase<TFloat>::operator+(TFloat scal) const
{
	Vector3DBase res;

	res.x = x + scal;
	res.y = y + scal;
	res.z = z + scal;

	return res;
}


template <typename TFloat>
Vector3DBase<TFloat> Vector3DBase<TFloat>::operator-(TFloat scal) const
{
	Vector3DBase res;

	res.x = x - scal;
	res.y = y - scal;
	res.z = z - scal;

	return res;
}


template <typename TFloat>
Vector3DBase<TFloat> Vector3DBase<TFloat>::operator*(TFloat scal) const
{
	Vector3DBase res;

	res.x = x * scal;
	res.y = y * scal;
	res.z = z * scal;

	return res;
}


template <typename TFloat>
Vector3DBase<TFloat> Vector3DBase<TFloat>::operator/(TFloat scal) const
{
	Vector3DBase res;

	if (fabs(scal) > DOUBLE_PRECISION)
	{
		res.x = x / scal;
		res.y = y / scal;
		res.z = z / scal;
	}
	else
	{
		res.x = LARGE_VALUE;
		res.y = LARGE_VALUE;
		res.z = LARGE_VALUE;
	}

	return res;
}

// return true if vectors are the same:
template <typename TFloat>
bool Vector3DBase<TFloat>::operator==(const Vector3DBase& vector) const
{
	Vector3DBase tmp;
	bool res;

	tmp.x = x - vector.x;
	tmp.y = y - vector.y;
	tmp.z = z - vector.z;
	if (tmp.getNorm() < DOUBLE_PRECISION)
	{
		res = true;
	}
	else
	{
		res = false;
	}

	return res;
}

template <typename TFloat>
template <typename TargetType>
Vector3DBase<TargetType> Vector3DBase<TFloat>::to() const
{
	return Vector3DBase<TargetType>{static_cast<TargetType>(x),
	                                static_cast<TargetType>(y),
	                                static_cast<TargetType>(z)};
}
template Vector3DBase<double> Vector3DBase<float>::to() const;
template Vector3DBase<float> Vector3DBase<double>::to() const;

template <typename TFloat>
std::ostream& operator<<(std::ostream& oss, const Vector3DBase<TFloat>& v)
{
	oss << "(" << v.x << ", " << v.y << ", " << v.z << ")";
	return oss;
}

template std::ostream& operator<<(std::ostream& oss,
                                  const Vector3DBase<double>& v);
template std::ostream& operator<<(std::ostream& oss,
                                  const Vector3DBase<float>& v);

template class Vector3DBase<double>;
template class Vector3DBase<float>;