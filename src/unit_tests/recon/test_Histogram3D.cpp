/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "datastruct/projection/Histogram3D.hpp"
#include "datastruct/projection/ListModeLUT.hpp"
#include "datastruct/scanner/GCDetRegular.hpp"
#include "utils/GCReconstructionUtils.hpp"

bool check_coords(std::array<coord_t, 3> c1, std::array<coord_t, 3> c2)
{
	return (c1[0] == c2[0] && c1[1] == c2[1] && c1[2] == c2[2]);
}
bool compare_coords(std::array<coord_t, 3> c1, std::array<coord_t, 3> c2)
{
	if (c1[2] < c2[2])
	{
		return true;
	}
	else if (c1[2] > c2[2])
	{
		return false;
	}
	else
	{
		if (c1[1] < c2[1])
		{
			return true;
		}
		else if (c1[1] > c2[1])
		{
			return false;
		}
		else
		{
			if (c1[0] < c2[0])
			{
				return true;
			}
			else if (c1[0] > c2[0])
			{
				return false;
			}
			else
			{
				// Fully equal
				return false;
			}
		}
	}
}

TEST_CASE("histo3d", "[histo]")
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

	size_t n_total_detectors =
	    scanner->num_doi * scanner->num_rings * scanner->dets_per_ring;
	auto histo3d = std::make_unique<Histogram3DOwned>(scanner.get());

	SECTION("histo3d-sizes")
	{
		REQUIRE(histo3d->n_r == 28);  // 7*(2*2)
		REQUIRE(histo3d->n_phi == 24);
		REQUIRE(histo3d->n_z_bin == 61);
	}

	SECTION("histo3d-binIds")
	{
		for (bin_t binId = 0; binId < histo3d->count(); binId++)
		{
			coord_t r, phi, z_bin;
			histo3d->getCoordsFromBinId(binId, r, phi, z_bin);
			bin_t supposedBin = histo3d->getBinIdFromCoords(r, phi, z_bin);
			REQUIRE(supposedBin == binId);
		}
	}

	SECTION("histo3d-coords-binning")
	{
		coord_t r, phi, z_bin;
		for (r = 0; r < histo3d->n_r; r++)
		{
			for (phi = 0; phi < histo3d->n_phi; phi++)
			{
				for (z_bin = 0; z_bin < histo3d->n_z_bin; z_bin++)
				{
					det_id_t d1, d2;
					coord_t r_supp, phi_supp, z_bin_supp;
					histo3d->getDetPairFromCoords(r, phi, z_bin, d1, d2);
					histo3d->getCoordsFromDetPair(d1, d2, r_supp, phi_supp,
					                              z_bin_supp);
					det_id_t d1_supp, d2_supp;
					histo3d->getDetPairFromCoords(r_supp, phi_supp, z_bin_supp,
					                              d1_supp, d2_supp);

					bool check = (d1_supp == d1 && d2_supp == d2) ||
					             (d1_supp == d2 && d2_supp == d1);
					REQUIRE(check);
					REQUIRE(d1 < n_total_detectors);
					REQUIRE(d2 < n_total_detectors);
				}
			}
		}
	}

	SECTION("histo3d-detector-swap")
	{
		coord_t r, phi, z_bin;
		for (r = 0; r < histo3d->n_r; r++)
		{
			for (phi = 0; phi < histo3d->n_phi; phi++)
			{
				for (z_bin = 0; z_bin < histo3d->n_z_bin; z_bin++)
				{
					det_id_t d1, d2;
					coord_t r_supp, phi_supp, z_bin_supp;
					histo3d->getDetPairFromCoords(r, phi, z_bin, d1, d2);
					histo3d->getCoordsFromDetPair(d2, d1, r_supp, phi_supp,
					                              z_bin_supp);
					REQUIRE(r_supp == r);
					REQUIRE(phi_supp == phi);
					REQUIRE(z_bin_supp == z_bin);
				}
			}
		}
	}


	SECTION("histo3d-detectors-uniqueness")
	{
		std::vector<std::array<coord_t, 3>> all_coords;

		for (det_id_t d1 = 0; d1 < n_total_detectors; d1++)
		{
			for (det_id_t d2 = d1 + 1; d2 < n_total_detectors; d2++)
			{
				int d1_ring = d1 % (scanner->dets_per_ring);
				int d2_ring = d2 % (scanner->dets_per_ring);
				int diff = std::abs(d1_ring - d2_ring);
				diff = (diff < static_cast<int>(scanner->dets_per_ring / 2)) ?
				           diff :
				           scanner->dets_per_ring - diff;
				if (diff < static_cast<int>(scanner->min_ang_diff))
				{
					continue;
				}
				int z1 = (d1 / (scanner->dets_per_ring)) % (scanner->num_rings);
				int z2 = (d2 / (scanner->dets_per_ring)) % (scanner->num_rings);
				if (std::abs(z1 - z2) > scanner->max_ring_diff)
				{
					continue;
				}

				coord_t r, phi, z_bin;
				histo3d->getCoordsFromDetPair(d1, d2, r, phi, z_bin);
				all_coords.push_back({r, phi, z_bin});
			}
		}
		std::sort(std::begin(all_coords), std::end(all_coords), compare_coords);
		auto u = std::unique(std::begin(all_coords), std::end(all_coords),
		                     check_coords);
		bool containsDuplicate = u != std::end(all_coords);
		REQUIRE(!containsDuplicate);
	}

	SECTION("histo3d-line-integrity")
	{
		auto someListMode = std::make_unique<ListModeLUTOwned>(scanner.get());
		double epsilon = 1e-5;
		someListMode->allocate(3);
		someListMode->setDetectorIdsOfEvent(0, 15, 105);
		someListMode->setDetectorIdsOfEvent(1, 238, 75);
		someListMode->setDetectorIdsOfEvent(2, 200, 110);
		for (size_t lmEv = 0; lmEv < someListMode->count(); lmEv++)
		{
			auto [d1, d2] = someListMode->getDetectorPair(lmEv);
			GCStraightLineParam lor = Util::getNativeLOR(*scanner, *someListMode, lmEv);
			bin_t binId = histo3d->getBinIdFromDetPair(d1, d2);
			GCStraightLineParam histoLor = Util::getNativeLOR(*scanner, *histo3d, binId);

			CHECK(((std::abs(histoLor.point1.x - lor.point1.x) < epsilon &&
			        std::abs(histoLor.point1.y - lor.point1.y) < epsilon &&
			        std::abs(histoLor.point1.z - lor.point1.z) < epsilon &&
			        std::abs(histoLor.point2.x - lor.point2.x) < epsilon &&
			        std::abs(histoLor.point2.y - lor.point2.y) < epsilon &&
			        std::abs(histoLor.point2.z - lor.point2.z) < epsilon) ||
			       (std::abs(histoLor.point1.x - lor.point2.x) < epsilon &&
			        std::abs(histoLor.point1.y - lor.point2.y) < epsilon &&
			        std::abs(histoLor.point1.z - lor.point2.z) < epsilon &&
			        std::abs(histoLor.point2.x - lor.point1.x) < epsilon &&
			        std::abs(histoLor.point2.y - lor.point1.y) < epsilon &&
			        std::abs(histoLor.point2.z - lor.point1.z) < epsilon)));
		}
	}

	SECTION("histo3d-bin-iterator")
	{
		size_t num_subsets = histo3d->n_phi;  // Have a subset for every angle
		for (size_t subset = 0; subset < num_subsets; subset++)
		{
			auto binIter = histo3d->getBinIter(num_subsets, subset);
			CHECK(binIter->size() == histo3d->count() / num_subsets);
			CHECK(binIter->size() == histo3d->n_r * histo3d->n_z_bin);
			coord_t r0, phi0, z_bin0;
			histo3d->getCoordsFromBinId(binIter->get(0), r0, phi0, z_bin0);
			for (bin_t bin = 1; bin < binIter->size(); bin++)
			{
				coord_t r, phi, z_bin;
				histo3d->getCoordsFromBinId(binIter->get(bin), r, phi, z_bin);
				REQUIRE(phi == phi0);
			}
		}
	}

	SECTION("histo3d-get-lor-id")
	{
		bin_t binId = 12;
		auto [d1_ref, d2_ref] = histo3d->getDetectorPair(binId);
		histo_bin_t lorId = histo3d->getHistogramBin(binId);
		auto newBin = std::get<bin_t>(lorId);
		det_pair_t detPair = histo3d->getDetectorPair(newBin);
		CHECK(d1_ref == detPair.d1);
		CHECK(d2_ref == detPair.d2);
	}
}
