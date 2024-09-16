/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/IListMode.hpp"
#include "utils/GCAssert.hpp"
#include "utils/GCTypes.hpp"

#include <stdexcept>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

void py_setup_ilistmode(py::module& m)
{
	auto c = py::class_<IListMode, IProjectionData>(m, "IListMode");
	c.def("getProjectionValue", &IListMode::getProjectionValue);
	c.def("setProjectionValue", &IListMode::setProjectionValue);
	c.def("getBinIter", &IListMode::getBinIter);
}

#endif  // if BUILD_PYBIND11

float IListMode::getProjectionValue(bin_t id) const
{
	(void) id;
	return 1.0f;
}

void IListMode::setProjectionValue(bin_t id, float val)
{
	(void) id;
	(void) val;
	throw std::logic_error("setProjectionValue unimplemented");
}

std::unique_ptr<BinIterator> IListMode::getBinIter(int numSubsets,
                                                     int idxSubset) const
{
	ASSERT_MSG(idxSubset < numSubsets,
	           "The subset index has to be smaller than the number of subsets");
	ASSERT_MSG(
	    idxSubset >= 0 && numSubsets > 0,
	    "The subset index cannot be negative, the number of subsets cannot "
	    "be less than or equal to zero");

	size_t numEvents = count();
	return std::make_unique<BinIteratorChronological>(numSubsets, numEvents,
	                                                    idxSubset);
}
