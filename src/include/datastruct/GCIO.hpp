/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/PluginFramework.hpp"
#include "datastruct/projection/IProjectionData.hpp"
#include "operators/GCOperatorProjector.hpp"

#include <memory>
#include <string>

class GCScanner;

namespace IO
{
	std::unique_ptr<IProjectionData> openProjectionData(
	    const std::string& input_fname, const std::string& input_format,
	    const GCScanner& scanner, const Plugin::OptionsResult&);

	std::string possibleFormats(Plugin::InputFormatsChoice choice = Plugin::InputFormatsChoice::ALL);

	bool isFormatListMode(const std::string& format);

	// Projector-related
	GCOperatorProjector::ProjectorType
	    getProjector(const std::string& projectorName);
	bool requiresGPU(GCOperatorProjector::ProjectorType projector);

}  // namespace IO
