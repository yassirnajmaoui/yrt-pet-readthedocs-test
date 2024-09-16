/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "datastruct/image/GCImage.hpp"
#include "datastruct/projection/GCListModeLUT.hpp"
#include "datastruct/projection/IListMode.hpp"
#include "datastruct/scanner/GCDetRegular.hpp"
#include "operators/GCOperatorProjectorDD.hpp"
#include "operators/GCOperatorProjectorSiddon.hpp"
#include "utils/GCReconstructionUtils.hpp"

#include <algorithm>
#include <utility>

#if BUILD_CUDA
#include "recon/GCOSEM_gpu.cuh"
#endif


double get_rmse(const GCImage* img_ref, const GCImage* img)
{
	const GCImageParams& params = img_ref->getParams();
	const size_t numPixels =
	    static_cast<size_t>(params.nx * params.ny * params.nz);
	const double* ptr_ref = img_ref->getData().getRawPointer();
	const double* ptr = img->getData().getRawPointer();
	double rmse = 0.0;

	for (size_t i = 0; i < numPixels; i++)
	{
		rmse += std::pow(ptr_ref[i] - ptr[i], 2.0);
	}

	rmse = std::sqrt(rmse / static_cast<double>(numPixels));

	return rmse;
}

void dd(const GCScanner* scanner, IListMode* proj,
        std::unique_ptr<GCImage>& out, const bool flag_cuda)
{
	const auto osem = Util::createOSEM(scanner, flag_cuda);
	osem->imageParams = out->getParams();
	osem->num_OSEM_subsets = 1;
	osem->setSensDataInput(proj);
	if (flag_cuda)
	{
		osem->projectorType = GCOperatorProjector::DD_GPU;
	}
	else
	{
		osem->projectorType = GCOperatorProjector::DD;
	}
	std::vector<std::unique_ptr<GCImage>> sensImages;
	osem->generateSensitivityImages(sensImages, "");
	out = std::move(sensImages[0]);
}

TEST_CASE("DD-simple", "[dd]")
{
	SECTION("get_overlap")
	{
		CHECK(GCOperatorProjectorDD::get_overlap(1.1, 4.1, 2.1, 3.1) ==
		      Approx(1.0));
		CHECK(GCOperatorProjectorDD::get_overlap(4, 1, 2, 3) == Approx(0.0));
		CHECK(GCOperatorProjectorDD::get_overlap(4.5, 2.3, 1.6, 3.2) ==
		      Approx(0.0));
		CHECK(GCOperatorProjectorDD::get_overlap(1.1, 1.2, 1.3, 1.4) ==
		      Approx(0.0));
		CHECK(GCOperatorProjectorDD::get_overlap(1.4, 1.3, 1.1, 1.2) ==
		      Approx(0.0));
		CHECK(GCOperatorProjectorDD::get_overlap(9.2, 10.9, 8.3, 10.0) ==
		      Approx(10.0 - 9.2));
		CHECK(GCOperatorProjectorDD::get_overlap(9.2, 9.9, 8.3, 10.0) ==
		      Approx(9.9 - 9.2));
	}
}

TEST_CASE("DD", "[dd]")
{
#if BUILD_CUDA
	srand(13);

	// Create Scanner
	auto scanner = std::make_unique<GCScannerAlias>();
	scanner->scannerRadius = 200;
	scanner->axialFOV = 100;
	scanner->dets_per_ring = 64;
	scanner->num_rings = 8;
	scanner->num_doi = 1;
	scanner->max_ring_diff = 9;
	scanner->min_ang_diff = 3;
	scanner->dets_per_block = 1;
	scanner->crystalDepth = 0.5;
	scanner->crystalSize_trans = 0.15;
	scanner->crystalSize_z = 0.15;
	auto detRegular = std::make_unique<GCDetRegular>(scanner.get());
	detRegular->generateLUT();
	scanner->setDetectorSetup(detRegular.get());
	size_t numDets =
	    scanner->num_doi * scanner->num_rings * scanner->dets_per_ring;

	// Create some image
	// Setup image
	const int nx = 100;
	const int ny = 100;
	const int nz = 100;
	const double sx = 256.0;
	const double sy = 256.0;
	const double sz = 96.0;
	const double ox = 0.0;
	const double oy = 0.0;
	const double oz = 0.0;
	GCImageParams img_params(nx, ny, nz, sx, sy, sz, ox, oy, oz);
	auto img = std::make_unique<GCImageOwned>(img_params);
	img->allocate();

	auto data = std::make_unique<GCListModeLUTOwned>(scanner.get());
	const size_t numEvents = 500;
	data->allocate(numEvents);
	for (bin_t binId = 0; binId < numEvents; binId++)
	{
		const det_id_t d1 = rand() % numDets;
		const det_id_t d2 = rand() % numDets;
		data->setDetectorIdsOfEvent(binId, d1, d2);
	}

	// Helpter aliases
	using GCImageUniquePTR = std::unique_ptr<GCImage>;
	const auto toOwned = [](const GCImageUniquePTR& i)
	{ return reinterpret_cast<GCImageOwned*>(i.get()); };

	GCImageUniquePTR img_cpu = std::make_unique<GCImageOwned>(img_params);
	toOwned(img_cpu)->allocate();
	img_cpu->setValue(0.0);
	dd(scanner.get(), data.get(), img_cpu, false);

	GCImageUniquePTR img_gpu = std::make_unique<GCImageOwned>(img_params);
	toOwned(img_gpu)->allocate();
	img_gpu->setValue(0.0);
	dd(scanner.get(), data.get(), img_gpu, true);

	const double rmseCpuGpu = get_rmse(img_gpu.get(), img_cpu.get());

	CHECK(rmseCpuGpu < 0.01);
#endif
}
