/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include <iostream>

void assertion_failed(char const* expr, char const* function, char const* file,
                      long line, bool critical)
{
	std::cerr << "Assertion failed: " << expr << "\n\tin function " << function
	          << " at " << file << ":" << line << std::endl;
	if(critical)
	{
		throw std::runtime_error("Critial assertion error");
	}
}
void assertion_failed_msg(char const* expr, char const* msg,
                          char const* function, char const* file, long line,
                          bool critical)
{
	std::cerr << "Assertion failed: " << expr << "\n\tin function " << function
	          << " at " << file << ":" << line << std::endl;
	std::string fullErrorMessage = "Critial assertion error: ";
	fullErrorMessage += msg;
	if(critical)
	{
		throw std::runtime_error(fullErrorMessage);
	}
	else
	{
		std::cerr << fullErrorMessage << std::endl;
	}
}
