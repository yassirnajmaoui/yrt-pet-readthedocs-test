/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/GCVector.hpp"

#include "geometry/GCConstants.hpp"

#include <cmath>

#if BUILD_PYBIND11
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <sstream>

namespace py = pybind11;

template <typename TFloat>
void py_setup_gcvectorbase(py::module& m)
{
	std::string className = "GCVector";
	if (typeid(TFloat) == typeid(float))
	{
		className += "Float";
	}

	auto c = py::class_<GCVectorBase<TFloat>>(m, className.c_str());
	c.def(py::init<>());
	c.def(py::init<TFloat, TFloat, TFloat>());
	c.def(py::init<const GCVectorBase<TFloat>&>());
	c.def("getNorm", &GCVectorBase<TFloat>::getNorm);
	c.def("update",
	      static_cast<void (GCVectorBase<TFloat>::*)(TFloat, TFloat, TFloat)>(
	          &GCVectorBase<TFloat>::update));
	c.def("update",
	      static_cast<void (GCVectorBase<TFloat>::*)(
	          const GCVectorBase<TFloat>&)>(&GCVectorBase<TFloat>::update));
	c.def("normalize", &GCVectorBase<TFloat>::normalize);
	c.def("__sub__",
	      static_cast<GCVectorBase<TFloat> (GCVectorBase<TFloat>::*)(
	          const GCVectorBase<TFloat>&) const>(
	          &GCVectorBase<TFloat>::operator-),
	      py::is_operator());
	c.def("__add__",
	      static_cast<GCVectorBase<TFloat> (GCVectorBase<TFloat>::*)(
	          const GCVectorBase<TFloat>&) const>(
	          &GCVectorBase<TFloat>::operator+),
	      py::is_operator());
	c.def("__mul__",
	      static_cast<GCVectorBase<TFloat> (GCVectorBase<TFloat>::*)(
	          const GCVectorBase<TFloat>&) const>(
	          &GCVectorBase<TFloat>::operator*),
	      py::is_operator());
	c.def("__sub__",
	      static_cast<GCVectorBase<TFloat> (GCVectorBase<TFloat>::*)(TFloat)
	                      const>(&GCVectorBase<TFloat>::operator-),
	      py::is_operator());
	c.def("__add__",
	      static_cast<GCVectorBase<TFloat> (GCVectorBase<TFloat>::*)(TFloat)
	                      const>(&GCVectorBase<TFloat>::operator+),
	      py::is_operator());
	c.def("__mul__",
	      static_cast<GCVectorBase<TFloat> (GCVectorBase<TFloat>::*)(TFloat)
	                      const>(&GCVectorBase<TFloat>::operator*),
	      py::is_operator());
	c.def("__div__",
	      static_cast<GCVectorBase<TFloat> (GCVectorBase<TFloat>::*)(TFloat)
	                      const>(&GCVectorBase<TFloat>::operator/),
	      py::is_operator());
	c.def(
	    "__eq__",
	    static_cast<bool (GCVectorBase<TFloat>::*)(const GCVectorBase<TFloat>&)
	                    const>(&GCVectorBase<TFloat>::operator==),
	    py::is_operator());
	c.def("__repr__",
	      [](const GCVectorBase<TFloat>& self)
	      {
		      std::stringstream ss;
		      ss << self;
		      return ss.str();
	      });
	c.def_readwrite("x", &GCVectorBase<TFloat>::x);
	c.def_readwrite("y", &GCVectorBase<TFloat>::y);
	c.def_readwrite("z", &GCVectorBase<TFloat>::z);
	c.def_readwrite("isNormalized", &GCVectorBase<TFloat>::isNormalized);
	c.def("getNormalized", &GCVectorBase<TFloat>::getNormalized);
}

void py_setup_gcvector(py::module& m)
{
	py_setup_gcvectorbase<float>(m);
	py_setup_gcvectorbase<double>(m);
}

#endif

template <typename TFloat>
GCVectorBase<TFloat>::GCVectorBase(TFloat xi, TFloat yi, TFloat zi)
    : x(xi), y(yi), z(zi), isNormalized(false)
{
}


template <typename TFloat>
GCVectorBase<TFloat>::GCVectorBase(const GCVectorBase& v)
{
	x = v.x;
	y = v.y;
	z = v.z;
	isNormalized = v.isNormalized;
}


template <typename TFloat>
GCVectorBase<TFloat>::GCVectorBase() : x(0.), y(0.), z(0.), isNormalized(false)
{
}


// destructor
template <typename TFloat>
GCVectorBase<TFloat>::~GCVectorBase() = default;

template <typename TFloat>
TFloat GCVectorBase<TFloat>::getNorm() const
{
	TFloat norm;

	norm = sqrt(x * x + y * y + z * z);

	return norm;
}


template <typename TFloat>
void GCVectorBase<TFloat>::update(TFloat xi, TFloat yi, TFloat zi)
{
	x = xi;
	y = yi;
	z = zi;
	isNormalized = false;
}


template <typename TFloat>
GCVectorBase<TFloat>& GCVectorBase<TFloat>::operator=(const GCVectorBase& v)
{
	x = v.x;
	y = v.y;
	z = v.z;
	isNormalized = v.isNormalized;

	return *this;
}


// update 3:
template <typename TFloat>
void GCVectorBase<TFloat>::update(const GCVectorBase& v)
{
	x = v.x;
	y = v.y;
	z = v.z;
	isNormalized = v.isNormalized;
}

template <typename TFloat>
GCVectorBase<TFloat>& GCVectorBase<TFloat>::normalize()
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
GCVectorBase<TFloat> GCVectorBase<TFloat>::getNormalized()
{
	TFloat norm;
	GCVectorBase res;

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
GCVectorBase<TFloat>
    GCVectorBase<TFloat>::operator-(const GCVectorBase& v) const
{
	GCVectorBase res;

	res.x = x - v.x;
	res.y = y - v.y;
	res.z = z - v.z;

	return res;
}


template <typename TFloat>
GCVectorBase<TFloat>
    GCVectorBase<TFloat>::operator+(const GCVectorBase& vector) const
{
	GCVectorBase res;

	res.x = x + vector.x;
	res.y = y + vector.y;
	res.z = z + vector.z;

	return res;
}


template <typename TFloat>
TFloat GCVectorBase<TFloat>::scalProd(const GCVectorBase& vector) const
{
	TFloat res;

	res = x * vector.x + y * vector.y + z * vector.z;

	return res;
}

template <typename TFloat>
GCVectorBase<TFloat>
    GCVectorBase<TFloat>::crossProduct(const GCVectorBase& B) const
{
	GCVectorBase P(0.0, 0.0, 0.0);
	P.x = this->y * B.z - this->z * B.y;
	P.y = this->z * B.x - this->x * B.z;
	P.z = this->x * B.y - this->y * B.x;
	return P;
}

template <typename TFloat>
void GCVectorBase<TFloat>::linear_transformation(const GCVectorBase& i,
                                                 const GCVectorBase& j,
                                                 const GCVectorBase& k)
{
	this->x = i.x * this->x + j.x * this->y + k.x * this->z;
	this->y = i.y * this->x + j.y * this->y + k.y * this->z;
	this->z = i.z * this->x + j.z * this->y + k.z * this->z;
}

template <typename TFloat>
int GCVectorBase<TFloat>::argmax()
{
	return (x > y) ? ((x > z) ? 0 : 2) : ((y > z) ? 1 : 2);
}

template <typename TFloat>
TFloat GCVectorBase<TFloat>::operator[](int idx) const
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
TFloat GCVectorBase<TFloat>::operator[](int idx)
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
GCVectorBase<TFloat>
    GCVectorBase<TFloat>::operator*(const GCVectorBase& vector) const
{
	GCVectorBase res;

	res.x = y * vector.z - z * vector.y;
	res.y = z * vector.x - x * vector.z;
	res.z = x * vector.y - y * vector.x;

	return res;
}


template <typename TFloat>
GCVectorBase<TFloat> GCVectorBase<TFloat>::operator+(TFloat scal) const
{
	GCVectorBase res;

	res.x = x + scal;
	res.y = y + scal;
	res.z = z + scal;

	return res;
}


template <typename TFloat>
GCVectorBase<TFloat> GCVectorBase<TFloat>::operator-(TFloat scal) const
{

	GCVectorBase res;

	res.x = x - scal;
	res.y = y - scal;
	res.z = z - scal;

	return res;
}


template <typename TFloat>
GCVectorBase<TFloat> GCVectorBase<TFloat>::operator*(TFloat scal) const
{
	GCVectorBase res;

	res.x = x * scal;
	res.y = y * scal;
	res.z = z * scal;

	return res;
}


template <typename TFloat>
GCVectorBase<TFloat> GCVectorBase<TFloat>::operator/(TFloat scal) const
{

	GCVectorBase res;

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
bool GCVectorBase<TFloat>::operator==(const GCVectorBase& vector) const
{
	GCVectorBase tmp;
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
std::ostream& operator<<(std::ostream& oss, const GCVectorBase<TFloat>& v)
{
	oss << "(" << v.x << ", " << v.y << ", " << v.z << ")";
	return oss;
}

template std::ostream& operator<<(std::ostream& oss,
                                  const GCVectorBase<double>& v);
template std::ostream& operator<<(std::ostream& oss,
                                  const GCVectorBase<float>& v);

template class GCVectorBase<double>;
template class GCVectorBase<float>;
