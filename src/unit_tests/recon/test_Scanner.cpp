/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "datastruct/projection/Histogram3D.hpp"
#include "datastruct/scanner/DetRegular.hpp"
#include "datastruct/scanner/Scanner.hpp"
#include "utils/Array.hpp"

TEST_CASE("scanner", "[createLUT]")
{
	auto scanner = std::make_unique<ScannerAlias>();  // Fake small scanner
	scanner->scannerRadius = 2;
	scanner->axialFOV = 200;
	scanner->dets_per_ring = 24;
	scanner->num_rings = 9;
	scanner->num_doi = 2;
	scanner->max_ring_diff = 4;
	scanner->min_ang_diff = 6;
	scanner->dets_per_block = 1;
	scanner->crystalDepth = 0.5;
	auto detRegular = std::make_unique<DetRegular>(scanner.get());
	detRegular->generateLUT();
	scanner->setDetectorSetup(detRegular.get());
	auto histo3d = std::make_unique<Histogram3DOwned>(*scanner);

	Array2D<float> lut;
	scanner->createLUT(lut);

	SECTION("lut-size")
	{
		REQUIRE(lut.getSize(0) == scanner->getNumDets());
		REQUIRE(lut.getSize(1) == 6);
	}

	SECTION("lut-det_pos")
	{
		size_t bin_id = 100;
		Vector3DFloat pos = scanner->getDetectorPos(bin_id);
		REQUIRE(lut[bin_id][0] == pos.x);
		REQUIRE(lut[bin_id][1] == pos.y);
		REQUIRE(lut[bin_id][2] == pos.z);
	}
}
