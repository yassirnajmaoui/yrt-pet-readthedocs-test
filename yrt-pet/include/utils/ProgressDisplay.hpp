/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "utils/Types.hpp"

namespace Util
{
	class ProgressDisplay
	{
	public:
		explicit ProgressDisplay(int64_t p_total = -1,
		                           int64_t p_increment = 20);
		void progress(int64_t newProgress);
		void setTotal(int64_t p_total);
		void reset();
		bool isEnabled() const;
		void enable();
		void disable();

	private:
		int64_t m_total;
		int64_t m_lastDisplayedPercentage;
		int64_t m_increment;
		bool m_enabled;
	};
}  // namespace Util
