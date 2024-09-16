/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */
#pragma once

#include "datastruct/projection/IProjectionData.hpp"

#include <memory>

class IListMode : public IProjectionData
{
public:
	~IListMode() override = default;

	static constexpr bool IsListMode() { return true; }

	float getProjectionValue(bin_t id) const override;
	void setProjectionValue(bin_t id, float val) override;

	std::unique_ptr<BinIterator> getBinIter(int numSubsets,
	                                          int idxSubset) const override;
};
