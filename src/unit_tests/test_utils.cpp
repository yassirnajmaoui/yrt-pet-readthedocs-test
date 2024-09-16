#include "test_utils.hpp"

std::pair<std::unique_ptr<GCScannerAlias>,
          std::unique_ptr<GCDetRegular>> TestUtils::makeScanner()
{
	auto scanner = std::make_unique<GCScannerAlias>();  // Fake small scanner
	scanner->scannerRadius = 2;
	scanner->axialFOV = 200;
	scanner->dets_per_ring = 24;
	scanner->num_rings = 9;
	scanner->num_doi = 2;
	scanner->max_ring_diff = 4;
	scanner->min_ang_diff = 6;
	scanner->dets_per_block = 1;
	scanner->crystalDepth = 0.5;
	auto detRegular = std::make_unique<GCDetRegular>(scanner.get());
	detRegular->generateLUT();
	scanner->setDetectorSetup(detRegular.get());
	return std::pair<std::unique_ptr<GCScannerAlias>,
	                 std::unique_ptr<GCDetRegular>>(std::move(scanner),
	                                                std::move(detRegular));
}
