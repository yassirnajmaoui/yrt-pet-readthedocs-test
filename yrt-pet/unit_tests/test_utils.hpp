/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/scanner/DetRegular.hpp"
#include "datastruct/scanner/Scanner.hpp"

namespace TestUtils
{
	std::unique_ptr<Scanner> makeScanner();
}
