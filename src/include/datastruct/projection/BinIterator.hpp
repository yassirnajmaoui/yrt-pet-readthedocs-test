/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "utils/GCTypes.hpp"

#include <memory>
#include <vector>


class BinIterator
{
public:
	BinIterator() {}
	virtual ~BinIterator() {}
	virtual bin_t begin() const = 0;
	virtual bin_t end() const = 0;
	bin_t get(bin_t idx) const;
	virtual size_t size() const = 0;

private:
	virtual bin_t getSafe(bin_t idx) const = 0;
};

class BinIteratorRange : public BinIterator
{
public:
	BinIteratorRange(bin_t num);
	BinIteratorRange(bin_t p_idxStart, bin_t p_idxEnd, bin_t p_idxStride = 1);
	BinIteratorRange(std::tuple<bin_t, bin_t, bin_t> info);
	bin_t begin() const override;
	bin_t end() const override;
	size_t size() const override;

protected:
	bin_t getSafe(bin_t idx) const override;
	static bin_t getIdxEnd(bin_t idxStart, bin_t idxEnd, bin_t stride);

protected:
	bin_t idxStart;
	bin_t idxEnd;
	bin_t idxStride;
};

class BinIteratorRange2D : public BinIterator
{
public:
	BinIteratorRange2D(bin_t p_idxStart, bin_t p_numSlices, bin_t p_sliceSize,
	                     bin_t p_idxStride);
	bin_t begin() const override;
	bin_t end() const override;
	size_t size() const override;

protected:
	bin_t getSafe(bin_t idx) const override;

protected:
	bin_t idxStart;
	bin_t numSlices;
	bin_t sliceSize;
	bin_t idxStride;
};

class BinIteratorRangeHistogram3D : public BinIterator
{
public:
	BinIteratorRangeHistogram3D(size_t p_n_z_bin, size_t p_n_phi,
	                              size_t p_n_r, int p_num_subsets,
	                              int p_idx_subset);
	bin_t begin() const override;
	bin_t end() const override;
	size_t size() const override;

protected:
	bin_t getSafe(bin_t idx) const override;

protected:
	size_t n_z_bin, n_phi, n_r;
	size_t histoSize;
	int num_subsets, idx_subset;
	bin_t phi_0, phi_stride, n_phi_subset;
};

class BinIteratorVector : public BinIterator
{
public:
	BinIteratorVector(std::unique_ptr<std::vector<bin_t>>& p_idxList);
	bin_t begin() const override;
	bin_t end() const override;
	size_t size() const override;

protected:
	std::unique_ptr<std::vector<bin_t>> idxList;
	bin_t getSafe(bin_t idx) const override;
};

class BinIteratorChronological : public BinIteratorRange
{
public:
	BinIteratorChronological(bin_t p_numSubsets, bin_t p_numEvents,
	                           bin_t p_idxSubset);

private:
	static std::tuple<bin_t, bin_t, bin_t>
	    getSubsetRange(bin_t numSubsets, bin_t numEvents, bin_t idxSubset);
};
