/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "PluginOptionsHelper.hpp"

#include "utils/Assert.hpp"

#include <cxxopts.hpp>

namespace PluginOptionsHelper
{
	Plugin::OptionsResult
	    convertPluginResultsToMap(const cxxopts::ParseResult& result)
	{
		Plugin::OptionsResult optionsMap;

		for (const auto& option : result.arguments())
		{
			const auto& key = option.key();
			try
			{
				optionsMap[key] = result[key].as<std::string>();
			}
			catch (const std::bad_cast&)
			{
				try
				{
					// Exception for boolean values
					const bool caughtBool = result[key].as<bool>();
					optionsMap[key] = caughtBool ? "1" : "0";
				}
				catch (const std::bad_cast&)
				{
					// pass
				}
			}
		}
		return optionsMap;
	}

	void fillOptionsFromPlugins(cxxopts::Options& options,
	                            Plugin::InputFormatsChoice choice)
	{
		const Plugin::OptionsList pluginOptions =
		    Plugin::PluginRegistry::instance().getAllOptions(choice);

		// Group the plugin options in case two (or more) plugins gave the
		// same options

		// Key: Option name, vector of pairs {format name, corresponding option
		// info}
		using OptionsListGrouped = std::unordered_map<
		    std::string,
		    std::vector<std::pair<std::string, Plugin::OptionInfo>>>;
		OptionsListGrouped pluginOptionsGrouped;
		for (auto& pluginOption : pluginOptions)
		{
			const Plugin::OptionsListPerPlugin& optionsListInCurrentPlugin =
			    pluginOption.second;
			for (const Plugin::OptionPerPlugin& option :
			     optionsListInCurrentPlugin)
			{
				const Plugin::OptionInfo& optionInfo = option.second;
				const std::string& optionName = option.first;
				auto pluginOption_it = pluginOptionsGrouped.find(optionName);

				if (pluginOption_it == pluginOptionsGrouped.end())
				{
					// non existant, create
					pluginOptionsGrouped[optionName] = {
					    {pluginOption.first, optionInfo}};
				}
				else
				{
					// preexistant, append to it
					pluginOptionsGrouped[optionName].emplace_back(
					    pluginOption.first, optionInfo);
				}
			}
		}

		auto adder = options.add_options("Input format");
		for (auto& pluginOptionGrouped : pluginOptionsGrouped)
		{
			const auto& listOfPluginsThatHaveCurrentOption =
			    pluginOptionGrouped.second;
			std::string optionHelp;
			int isBool = -1;
			const size_t numPluginsThatHaveCurrentOptions =
			    listOfPluginsThatHaveCurrentOption.size();
			for (size_t i = 0; i < numPluginsThatHaveCurrentOptions; ++i)
			{
				const auto& [pluginName, helpForPlugin] =
				    listOfPluginsThatHaveCurrentOption[i];
				const bool isLastPlugin =
				    i == numPluginsThatHaveCurrentOptions - 1;

				optionHelp +=
				    "For " + pluginName + ": " + std::get<0>(helpForPlugin);
				if (!isLastPlugin)
				{
					optionHelp += "\n";
				}

				// It should not be allowed to provide two plugins that
				// have different IsBool (OptionInfo::second).
				// Send a warning here if that happens
				const int currentIsBool = (std::get<1>(helpForPlugin)) ? 1 : 0;
				if (isBool == -1)
				{
					// First init
					isBool = currentIsBool;
				}
				else
				{
					ASSERT_MSG(isBool == currentIsBool,
					           "A plugin already uses that option with a "
					           "different bool status");
				}
			}

			if (isBool == 1)
			{
				// parse boolean value
				adder(pluginOptionGrouped.first, optionHelp,
				      cxxopts::value<bool>());
			}
			else
			{
				// parse string value
				adder(pluginOptionGrouped.first, optionHelp,
				      cxxopts::value<std::string>());
			}
		}
	}
}  // namespace PluginOptionsHelper
