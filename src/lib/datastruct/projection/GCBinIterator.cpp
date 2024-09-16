/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/GCBinIterator.hpp"

#include <stdexcept>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
void py_setup_gcbiniterator(py::module& m)
{
	auto c = py::class_<GCBinIterator>(m, "GCBinIterator");
	c.def("get", &GCBinIterator::get);
	c.def("begin", &GCBinIterator::begin);
	c.def("end", &GCBinIterator::end);
	c.def("size", &GCBinIterator::size);

	auto c_range =
	    py::class_<GCBinIteratorRange, GCBinIterator>(m, "GCBinIteratorRange");
	c_range.def(py::init<bin_t>());
	c_range.def(py::init<bin_t, bin_t, bin_t>());  // add default argument
	c_range.def(py::init<std::tuple<bin_t, bin_t, bin_t>>());

	auto c_vector = py::class_<GCBinIteratorVector, GCBinIterator>(
	    m, "GCBinIteratorVector");
	c_vector.def(py::init(
	    [](const std::vector<bin_t>& vec)
	    {
		    auto idxs = std::make_unique<std::vector<bin_t>>(vec);
		    return GCBinIteratorVector(idxs);
	    }));

	auto c_chronological =
	    py::class_<GCBinIteratorChronological, GCBinIteratorRange>(
	        m, "GCBinIteratorChronological");
	c_chronological.def(py::init<bin_t, bin_t, bin_t>());
}
#endif

bin_t GCBinIterator::get(bin_t idx) const
{
	if (idx >= size())
	{
		throw std::range_error(
		    "The idx given does not exist in the range of the BinIterator");
	}
	return getSafe(idx);
}

GCBinIteratorRange::GCBinIteratorRange(bin_t num)
    : idxStart(0), idxEnd(num - 1), idxStride(1)
{
}

GCBinIteratorRange::GCBinIteratorRange(bin_t p_idxStart, bin_t p_idxEnd,
                                       bin_t p_idxStride)
    : idxStart(p_idxStart),
      idxEnd(getIdxEnd(p_idxStart, p_idxEnd, p_idxStride)),
      idxStride(p_idxStride)
{
}

GCBinIteratorRange::GCBinIteratorRange(std::tuple<bin_t, bin_t, bin_t> info)
    : idxStart(std::get<0>(info)),
      idxEnd(
          getIdxEnd(std::get<0>(info), std::get<1>(info), std::get<2>(info))),
      idxStride(std::get<2>(info))
{
}

bin_t GCBinIteratorRange::getIdxEnd(bin_t idxStart, bin_t idxEnd, bin_t stride)
{
	return idxStart + stride * ((idxEnd - idxStart) / stride);
}

bin_t GCBinIteratorRange::begin() const
{
	return idxStart;
}

bin_t GCBinIteratorRange::end() const
{
	return idxEnd;
}

bin_t GCBinIteratorRange::getSafe(bin_t idx) const
{
	return idxStart + idxStride * idx;
}

size_t GCBinIteratorRange::size() const
{
	return (idxEnd - idxStart) / idxStride + 1;
}

GCBinIteratorRange2D::GCBinIteratorRange2D(bin_t p_idxStart, bin_t p_numSlices,
                                           bin_t p_sliceSize, bin_t p_idxStride)
    : idxStart(p_idxStart),
      numSlices(p_numSlices),
      sliceSize(p_sliceSize),
      idxStride(p_idxStride)
{
}

bin_t GCBinIteratorRange2D::begin() const
{
	return idxStart;
}
bin_t GCBinIteratorRange2D::end() const
{
	return idxStart + numSlices * idxStride;
}
size_t GCBinIteratorRange2D::size() const
{
	return numSlices * sliceSize - 1;
}

bin_t GCBinIteratorRange2D::getSafe(bin_t idx) const
{
	bin_t sliceIdx = idx / sliceSize;
	bin_t idxOffset = idx % sliceSize;
	return idxStart + idxStride * sliceIdx + idxOffset;
}

GCBinIteratorRangeHistogram3D::GCBinIteratorRangeHistogram3D(size_t p_n_z_bin,
                                                             size_t p_n_phi,
                                                             size_t p_n_r,
                                                             int p_num_subsets,
                                                             int p_idx_subset)
    : n_z_bin(p_n_z_bin),
      n_phi(p_n_phi),
      n_r(p_n_r),
      num_subsets(p_num_subsets),
      idx_subset(p_idx_subset)
{
	phi_stride = num_subsets;
	phi_0 = idx_subset;
	n_phi_subset = n_phi / num_subsets;  // Number of rs in the subset
	// In the case that we would miss some bins because of the "floor" division
	// above
	if (phi_0 + n_phi_subset * phi_stride < n_phi)
	{
		n_phi_subset += 1;
	}
	histoSize = n_r * n_phi_subset * n_z_bin;
}

bin_t GCBinIteratorRangeHistogram3D::begin() const
{
	bin_t r = 0;
	bin_t phi = phi_0;
	bin_t z_bin = 0;
	return z_bin * n_phi * n_r + phi * n_r + r;
}

bin_t GCBinIteratorRangeHistogram3D::end() const
{
	bin_t r = n_r - 1;
	bin_t phi = (phi_stride * (n_phi_subset - 1)) + phi_0;
	bin_t z_bin = (n_z_bin - 1);
	return z_bin * n_phi * n_r + phi * n_r + r;
}

size_t GCBinIteratorRangeHistogram3D::size() const
{
	return histoSize;
}

bin_t GCBinIteratorRangeHistogram3D::getSafe(bin_t idx) const
{
	bin_t z_bin = idx / (n_phi_subset * n_r);
	bin_t phi = (idx % (n_phi_subset * n_r)) / n_r;
	bin_t r = (idx % (n_phi_subset * n_r)) % n_r;
	phi = phi_stride * phi + phi_0;  // scale and shift the phi coordinate
	return z_bin * n_phi * n_r + phi * n_r + r;
}


GCBinIteratorVector::GCBinIteratorVector(
    std::unique_ptr<std::vector<bin_t>>& p_idxList)
{
	idxList = std::move(p_idxList);
}

bin_t GCBinIteratorVector::begin() const
{
	return (*idxList.get())[0];
}

bin_t GCBinIteratorVector::end() const
{
	return (*idxList.get())[idxList->size() - 1];
}

bin_t GCBinIteratorVector::getSafe(bin_t idx) const
{
	return (*idxList.get())[idx];
}

size_t GCBinIteratorVector::size() const
{
	return idxList->size();
}


GCBinIteratorChronological::GCBinIteratorChronological(bin_t p_numSubsets,
                                                       bin_t p_numEvents,
                                                       bin_t p_idxSubset)
    : GCBinIteratorRange(
          getSubsetRange(p_numSubsets, p_numEvents, p_idxSubset))
{
}

std::tuple<bin_t, bin_t, bin_t>
    GCBinIteratorChronological::getSubsetRange(bin_t numSubsets,
                                               bin_t numEvents, bin_t idxSubset)
{
	if (idxSubset > numSubsets)
	{
		throw std::invalid_argument("The number of subsets has to be higher "
		                            "than the desired subset index.");
	}
	bin_t rest = numEvents % numSubsets;

	bin_t idxStart = ((numEvents - rest) * idxSubset) / numSubsets;
	bin_t idxEnd;

	if (idxSubset == numSubsets - 1)
	{
		// the last numBins % numSubsets are added here
		idxEnd =
		    (((numEvents - rest) * (idxSubset + 1)) / numSubsets + rest) - 1;
	}
	else
	{
		idxEnd = (((numEvents - rest) * (idxSubset + 1)) / numSubsets) - 1;
	}
	return std::make_tuple(idxStart, idxEnd, 1);
}
