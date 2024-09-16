/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/scanner/GCScanner.hpp"
#include "datastruct/scanner/GCDetRegular.hpp"

namespace TestUtils
{
	std::pair<std::unique_ptr<GCScannerAlias>,
	          std::unique_ptr<GCDetRegular>> makeScanner();
}
