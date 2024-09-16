/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/GCLORMotion.hpp"
#include "utils/GCAssert.hpp"

GCLORMotion::GCLORMotion(const std::string& filename)
    : m_structure{filename, 1 << 20}
{
	ASSERT_MSG(getNumFrames() > 0, "There has to be at least one frame");
	mp_startingTimestamps = m_structure.get_pointer<0>();
	mp_transforms = m_structure.get_pointer<1>();
}

transform_t GCLORMotion::getTransform(frame_t frame) const
{
	return mp_transforms[frame];
}

timestamp_t GCLORMotion::getStartingTimestamp(frame_t frame) const
{
	return mp_startingTimestamps[frame];
}

size_t GCLORMotion::getNumFrames() const
{
	return m_structure.get_num_columns();
}
