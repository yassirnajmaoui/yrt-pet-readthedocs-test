/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/fancyarray/trivial_struct_of_arrays.hpp"
#include "utils/GCTypes.hpp"

class GCLORMotion
{
public:
	GCLORMotion(const std::string& filename);

	transform_t getTransform(frame_t frame) const;
	timestamp_t getStartingTimestamp(frame_t frame) const;
	size_t getNumFrames() const;

private:
	using LORMotionStructure =
	    fancyarray::trivial_struct_of_arrays<timestamp_t, transform_t>;
	LORMotionStructure m_structure;

	const transform_t* mp_transforms;
	const timestamp_t* mp_startingTimestamps;
};
