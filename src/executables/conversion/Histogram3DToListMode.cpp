/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/GCHistogram3D.hpp"
#include "datastruct/projection/GCListModeLUT.hpp"
#include "datastruct/scanner/GCScanner.hpp"
#include "utils/GCGlobals.hpp"
#include "utils/GCReconstructionUtils.hpp"

#include <cxxopts.hpp>
#include <iostream>

int main(int argc, char** argv)
{
	std::string scanner_fname;
	std::string input_fname;
	std::string out_fname;
	size_t numEvents = 0;
	int numThreads = -1;

	// Parse command line arguments
	try
	{
		cxxopts::Options options(
		    argv[0], "Histogram3D to ListMode conversion driver");
		options.positional_help("[optional args]").show_positional_help();
		/* clang-format off */
		options.add_options()
		("s,scanner", "Scanner parameters file name",
		 cxxopts::value<std::string>(scanner_fname))
		("i,input", "Input histogram file",
		 cxxopts::value<std::string>(input_fname))
		("o,out", "Output list-mode filename",
		 cxxopts::value<std::string>(out_fname))
		("n,num", "Number of list-mode events",
		 cxxopts::value<size_t>(numEvents))
		("num_threads", "Number of threads to use",
		 cxxopts::value<int>(numThreads))
		("h,help", "Print help");
		/* clang-format on */
		auto result = options.parse(argc, argv);
		if (result.count("help"))
		{
			std::cout << options.help() << std::endl;
			return -1;
		}

		std::vector<std::string> required_params = {"scanner", "input", "out"};
		bool missing_args = false;
		for (auto& p : required_params)
		{
			if (result.count(p) == 0)
			{
				std::cerr << "Argument '" << p << "' missing" << std::endl;
				missing_args = true;
			}
		}
		if (missing_args)
		{
			std::cerr << options.help() << std::endl;
			return -1;
		}
	}
	catch (const cxxopts::exceptions::exception& e)
	{
		std::cerr << "Error parsing options: " << e.what() << std::endl;
		return -1;
	}

	GCGlobals::set_num_threads(numThreads);

	const auto scanner = std::make_unique<GCScannerOwned>(scanner_fname);
	auto histo =
	    std::make_unique<GCHistogram3DOwned>(scanner.get(), input_fname);

	const auto lm = std::make_unique<GCListModeLUTOwned>(scanner.get());
	Util::histogram3DToListModeLUT(histo.get(), lm.get(), numEvents);
	lm->writeToFile(out_fname);

	std::cout << "Done." << std::endl;
	return 0;
}
