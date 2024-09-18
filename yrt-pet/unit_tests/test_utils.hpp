/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/scanner/Scanner.hpp"
#include "datastruct/scanner/DetRegular.hpp"

namespace TestUtils
{
	std::pair<std::unique_ptr<ScannerAlias>,
	          std::unique_ptr<DetRegular>> makeScanner();
}
