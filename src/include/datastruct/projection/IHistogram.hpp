/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */
#pragma once

#include "datastruct/projection/IProjectionData.hpp"

class IHistogram : public IProjectionData
{
public:
	static constexpr bool IsListMode() { return false; }

	virtual float
	    getProjectionValueFromHistogramBin(histo_bin_t histoBinId) const = 0;
};
