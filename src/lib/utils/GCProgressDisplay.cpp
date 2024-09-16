/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/GCProgressDisplay.hpp"
#include <iostream>

namespace Util
{

	GCProgressDisplay::GCProgressDisplay(int64_t p_total, int64_t p_increment)
	    : m_total(p_total),
	      m_lastDisplayedPercentage(-1),
	      m_increment(p_increment),
	      m_enabled(true)
	{
	}

	void GCProgressDisplay::progress(int64_t newProgress)
	{
		if (isEnabled() && m_total > 0)
		{
			int64_t newPercentage = (100 * newProgress) / m_total;
			if (newPercentage >= m_lastDisplayedPercentage + m_increment ||
			    m_lastDisplayedPercentage < 0)
			{
				m_lastDisplayedPercentage = newPercentage;
				std::cout << "Progress: " << newPercentage << "%" << std::endl;
			}
		}
	}

	void GCProgressDisplay::setTotal(int64_t p_total)
	{
		m_total = p_total;
	}

	void GCProgressDisplay::reset()
	{
		m_lastDisplayedPercentage = -1;
	}

	bool GCProgressDisplay::isEnabled() const
	{
		return m_enabled;
	}

	void GCProgressDisplay::enable()
	{
		m_enabled = true;
	}

	void GCProgressDisplay::disable()
	{
		m_enabled = false;
	}

}  // namespace Util
