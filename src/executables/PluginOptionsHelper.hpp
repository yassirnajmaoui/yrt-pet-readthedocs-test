/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/GCPluginFramework.hpp"

#include <cxxopts.hpp>

namespace PluginOptionsHelper
{
	// Convert cxxopts's options to unordered map
	Plugin::OptionsResult
	    convertPluginResultsToMap(const cxxopts::ParseResult& result);

	void fillOptionsFromPlugins(
	    cxxopts::Options& options,
	    Plugin::InputFormatsChoice choice = Plugin::InputFormatsChoice::ALL);
}  // namespace PluginOptionsHelper
