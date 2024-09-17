/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/GCReconstructionUtils.hpp"

#include "datastruct/IO.hpp"
#include "datastruct/projection/Histogram3D.hpp"
#include "datastruct/projection/ListModeLUT.hpp"
#include "geometry/Matrix.hpp"
#include "operators/GCOperatorProjectorDD.hpp"
#include "operators/GCOperatorProjectorSiddon.hpp"
#include "recon/GCOSEM_cpu.hpp"
#include "utils/GCAssert.hpp"
#include "utils/GCGlobals.hpp"
#include "utils/GCProgressDisplay.hpp"


#if BUILD_CUDA
#include "operators/GCOperatorProjectorDD_gpu.cuh"
#include "recon/GCOSEM_gpu.cuh"
#endif


#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;

void py_setup_gcreconstructionutils(pybind11::module& m)
{
	m.def("histogram3DToListModeLUT", &Util::histogram3DToListModeLUT);
	m.def(
		"convertToHistogram3D",
		[](const Histogram& dat, Histogram3D& histoOut)
		{
			Util::convertToHistogram3D<false>(dat, histoOut);
		},
		py::arg("histogramDataInput"), py::arg("histoOut"));
	m.def(
		"convertToHistogram3D",
		[](const ListMode& dat, Histogram3D& histoOut)
		{
			Util::convertToHistogram3D<true>(dat, histoOut);
		},
		py::arg("listmodeDataInput"), py::arg("histoOut"));
	m.def("createOSEM", &Util::createOSEM, py::arg("scanner"),
	      py::arg("useGPU") = false);
	m.def("generateTORRandomDOI", &Util::generateTORRandomDOI,
	      py::arg("scanner"), py::arg("d1"), py::arg("d2"), py::arg("vmax"));

	m.def("forwProject",
	      static_cast<void (*)(const Scanner* scanner, const Image* img,
	                           ProjectionData* projData,
	                           GCOperatorProjector::ProjectorType projectorType,
	                           const Image* attImage,
	                           const Histogram* additiveHistogram)>(
		      &Util::forwProject),
	      py::arg("scanner"), py::arg("img"), py::arg("projData"),
	      py::arg("projectorType") = GCOperatorProjector::ProjectorType::SIDDON,
	      py::arg("attImage") = nullptr,
	      py::arg("additiveHistogram") = nullptr);
	m.def("forwProject",
	      static_cast<void (*)(
		      const Scanner* scanner, const Image* img,
		      ProjectionData* projData, const BinIterator& binIterator,
		      GCOperatorProjector::ProjectorType projectorType,
		      const Image* attImage, const Histogram* additiveHistogram)>(
		      &Util::forwProject),
	      py::arg("scanner"), py::arg("img"), py::arg("projData"),
	      py::arg("binIterator"),
	      py::arg("projectorType") = GCOperatorProjector::SIDDON,
	      py::arg("attImage") = nullptr,
	      py::arg("additiveHistogram") = nullptr);
	m.def("forwProject",
	      static_cast<void (*)(const Image* img, ProjectionData* projData,
	                           const GCOperatorProjectorParams& projParams,
	                           GCOperatorProjector::ProjectorType projectorType,
	                           const Image* attImage,
	                           const Histogram* additiveHistogram)>(
		      &Util::forwProject),
	      py::arg("img"), py::arg("projData"), py::arg("projParams"),
	      py::arg("projectorType") = GCOperatorProjector::SIDDON,
	      py::arg("attImage") = nullptr,
	      py::arg("additiveHistogram") = nullptr);

	m.def("backProject",
	      static_cast<void (*)(const Scanner* scanner, Image* img,
	                           const ProjectionData* projData,
	                           GCOperatorProjector::ProjectorType projectorType,
	                           const Image* attImage,
	                           const Histogram* additiveHistogram)>(
		      &Util::backProject),
	      py::arg("scanner"), py::arg("img"), py::arg("projData"),
	      py::arg("projectorType") = GCOperatorProjector::ProjectorType::SIDDON,
	      py::arg("attImage") = nullptr,
	      py::arg("additiveHistogram") = nullptr);
	m.def("backProject",
	      static_cast<void (*)(
		      const Scanner* scanner, Image* img,
		      const ProjectionData* projData, const BinIterator& binIterator,
		      GCOperatorProjector::ProjectorType projectorType,
		      const Image* attImage, const Histogram* additiveHistogram)>(
		      &Util::backProject),
	      py::arg("scanner"), py::arg("img"), py::arg("projData"),
	      py::arg("binIterator"),
	      py::arg("projectorType") = GCOperatorProjector::SIDDON,
	      py::arg("attImage") = nullptr,
	      py::arg("additiveHistogram") = nullptr);
	m.def("backProject",
	      static_cast<void (*)(Image* img, const ProjectionData* projData,
	                           const GCOperatorProjectorParams& projParams,
	                           GCOperatorProjector::ProjectorType projectorType,
	                           const Image* attImage,
	                           const Histogram* additiveHistogram)>(
		      &Util::backProject),
	      py::arg("img"), py::arg("projData"), py::arg("projParams"),
	      py::arg("projectorType") = GCOperatorProjector::SIDDON,
	      py::arg("attImage") = nullptr,
	      py::arg("additiveHistogram") = nullptr);
}

#endif

namespace Util
{
	void histogram3DToListModeLUT(const Histogram3D* histo,
	                              ListModeLUTOwned* lmOut, size_t numEvents)
	{
		ASSERT(lmOut != nullptr);
		const float* dataPtr = histo->getData().getRawPointer();

		// Phase 1: calculate sum of histogram values
		double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
		for (bin_t binId = 0; binId < histo->count(); binId++)
		{
			sum += dataPtr[binId];
		}

		// Default target number of events (histogram sum)
		if (numEvents == 0)
		{
			numEvents = std::lround(sum);
		}
		// Phase 2: calculate actual number of events
		size_t sumInt = 0.0;
#pragma omp parallel for reduction(+ : sumInt)
		for (bin_t binId = 0; binId < histo->count(); binId++)
		{
			sumInt += std::lround(dataPtr[binId] / sum * (double)numEvents);
		}

		// Allocate list-mode data
		lmOut->allocate(sumInt);

		int numThreads = GCGlobals::get_num_threads();
		if (numThreads > 1)
		{
			size_t numBinsPerThread =
				std::ceil(double(histo->count()) / (double)numThreads);
			Array1D<size_t> partialSums;
			partialSums.allocate(numThreads);

			// Phase 3: prepare partial sums for parallelization
#pragma omp parallel for num_threads(numThreads)
			for (int ti = 0; ti < numThreads; ti++)
			{
				bin_t binStart = ti * numBinsPerThread;
				bin_t binEnd = std::min(histo->count() - 1,
				                        binStart + numBinsPerThread - 1);
				for (bin_t binId = binStart; binId <= binEnd; binId++)
				{
					partialSums[ti] +=
						std::lround(dataPtr[binId] / sum * (double)numEvents);
				}
			}

			// Calculate indices
			Array1D<size_t> lmStartIdx;
			lmStartIdx.allocate(numThreads);
			lmStartIdx[0] = 0;
			for (int ti = 1; ti < numThreads; ti++)
			{
				lmStartIdx[ti] = lmStartIdx[ti - 1] + partialSums[ti - 1];
			}

			// Phase 4: create events by block
#pragma omp parallel for num_threads(numThreads)
			for (int ti = 0; ti < numThreads; ti++)
			{
				bin_t binStart = ti * numBinsPerThread;
				bin_t binEnd = std::min(histo->count() - 1,
				                        binStart + numBinsPerThread - 1);
				bin_t eventId = lmStartIdx[ti];
				for (bin_t binId = binStart; binId <= binEnd; binId++)
				{
					if (dataPtr[binId] != 0.f)
					{
						auto [d1, d2] = histo->getDetectorPair(binId);
						int numEventsBin = std::lround(dataPtr[binId] / sum *
							(double)numEvents);
						for (int ei = 0; ei < numEventsBin; ei++)
						{
							lmOut->setDetectorIdsOfEvent(eventId++, d1, d2);
						}
					}
				}
			}
		}
		else
		{
			bin_t eventId = 0;
			for (bin_t binId = 0; binId < histo->count(); binId++)
			{
				if (dataPtr[binId] != 0.f)
				{
					auto [d1, d2] = histo->getDetectorPair(binId);
					int numEventsBin =
						std::lround(dataPtr[binId] / sum * (double)numEvents);
					for (int ei = 0; ei < numEventsBin; ei++)
					{
						lmOut->setDetectorIdsOfEvent(eventId++, d1, d2);
					}
				}
			}
		}
	}

	template <bool RequiresAtomicAccumulation>
	void convertToHistogram3D(const ProjectionData& dat,
	                          Histogram3D& histoOut)
	{
		float* histoDataPointer = histoOut.getData().getRawPointer();
		const size_t numDatBins = dat.count();

		GCProgressDisplay progressBar(numDatBins, 1);
		int64_t currentProgress = 0;

		const Histogram3D* histoOut_constptr = &histoOut;
		const ProjectionData* dat_constptr = &dat;
#pragma omp parallel for default(none)                                         \
    firstprivate(histoDataPointer, progressBar, numDatBins, histoOut_constptr, \
                     dat_constptr) shared(currentProgress)
		for (bin_t datBin = 0; datBin < numDatBins; ++datBin)
		{
			// Still use atomic for the progress report
#pragma omp atomic
			++currentProgress;

			if (omp_get_thread_num() == 0)
			{
				progressBar.progress(currentProgress);
			}

			const float projValue = dat_constptr->getProjectionValue(datBin);
			if (projValue > 0)
			{
				const auto [d1, d2] = dat_constptr->getDetectorPair(datBin);
				if (d1 == d2)
				{
					// Do not crash
					continue;
				}
				const bin_t histoBin =
					histoOut_constptr->getBinIdFromDetPair(d1, d2);
				if constexpr (RequiresAtomicAccumulation)
				{
#pragma omp atomic
					histoDataPointer[histoBin] += projValue;
				}
				else
				{
					histoDataPointer[histoBin] += projValue;
				}
			}
		}
	}

	template void convertToHistogram3D<true>(const ProjectionData&,
	                                         Histogram3D&);
	template void convertToHistogram3D<false>(const ProjectionData&,
	                                          Histogram3D&);

	GCOperatorProjectorBase::ProjectionProperties
		getProjectionProperties(const Scanner& scanner,
		                        const ProjectionData& dat, bin_t bin)
	{
		auto [d1, d2] = dat.getDetectorPair(bin);

		StraightLineParam lor;
		if (dat.hasArbitraryLORs())
		{
			const line_t lorPts = dat.getArbitraryLOR(bin);
			lor =
				StraightLineParam{Vector3D{lorPts.x1, lorPts.y1, lorPts.z1},
				                  Vector3D{lorPts.x2, lorPts.y2, lorPts.z2}};
		}
		else
		{
			const Vector3DFloat p1 = scanner.getDetectorPos(d1);
			const Vector3DFloat p2 = scanner.getDetectorPos(d2);
			lor = StraightLineParam{p1, p2};
		}

		float tofValue = 0.0f;
		if (dat.hasTOF())
		{
			tofValue = dat.getTOFValue(bin);
		}
		float randomsEstimate = dat.getRandomsEstimate(bin);
		if (dat.hasMotion())
		{
			frame_t frame = dat.getFrame(bin);
			transform_t transfo = dat.getTransformOfFrame(frame);
			Vector3D point1 = lor.point1;
			Vector3D point2 = lor.point2;

			Matrix MRot(transfo.r00, transfo.r01, transfo.r02, transfo.r10,
			            transfo.r11, transfo.r12, transfo.r20, transfo.r21,
			            transfo.r22);
			Vector3D VTrans(transfo.tx, transfo.ty, transfo.tz);

			Vector3D point1Prim = MRot * point1;
			point1Prim = point1Prim + VTrans;

			Vector3D point2Prim = MRot * point2;
			point2Prim = point2Prim + VTrans;

			lor.update(point1Prim, point2Prim);
		}
		Vector3DFloat det1Orient = scanner.getDetectorOrient(d1);
		Vector3DFloat det2Orient = scanner.getDetectorOrient(d2);
		return GCOperatorProjectorBase::ProjectionProperties{
			lor, tofValue, randomsEstimate, det1Orient.to<double>(),
			det2Orient.to<double>()};
	}

	StraightLineParam getNativeLOR(const Scanner& scanner,
	                               const ProjectionData& dat, bin_t binId)
	{
		const auto [d1, d2] = dat.getDetectorPair(binId);
		const Vector3DFloat p1 = scanner.getDetectorPos(d1);
		const Vector3DFloat p2 = scanner.getDetectorPos(d2);
		return StraightLineParam{p1, p2};
	}

	std::tuple<StraightLineParam, Vector3D, Vector3D>
		generateTORRandomDOI(const Scanner* scanner, det_id_t d1, det_id_t d2,
		                     int vmax)
	{
		const Vector3DFloat p1 = scanner->getDetectorPos(d1);
		const Vector3DFloat p2 = scanner->getDetectorPos(d2);
		const Vector3DFloat n1 = scanner->getDetectorOrient(d1);
		const Vector3DFloat n2 = scanner->getDetectorOrient(d2);
		const double doi_1_t = (rand() % vmax) /
		                       (static_cast<float>(1 << 8) - 1) *
		                       scanner->crystalDepth;
		const double doi_2_t = (rand() % vmax) /
		                       (static_cast<float>(1 << 8) - 1) *
		                       scanner->crystalDepth;
		const Vector3D p1_doi(p1.x + doi_1_t * n1.x, p1.y + doi_1_t * n1.y,
		                      p1.z + doi_1_t * n1.z);
		const Vector3D p2_doi(p2.x + doi_2_t * n2.x, p2.y + doi_2_t * n2.y,
		                      p2.z + doi_2_t * n2.z);
		const StraightLineParam lorDOI(p1_doi, p2_doi);
		return {lorDOI, n1.to<double>(), n2.to<double>()};
	}

	std::unique_ptr<GCOSEM> createOSEM(const Scanner* scanner, bool useGPU)
	{
		std::unique_ptr<GCOSEM> osem;
		if (useGPU)
		{
#if BUILD_CUDA
			osem = std::make_unique<GCOSEM_gpu>(scanner);
#else
			throw std::runtime_error(
			    "Project not compiled with CUDA, GPU projectors unavailable.");
#endif
		}
		else
		{
			osem = std::make_unique<GCOSEM_cpu>(scanner);
		}
		return osem;
	}

	// Forward and backward projections
	template <bool IS_FWD>
	static void project(Image* img, ProjectionData* projData,
	                    const GCOperatorProjectorParams& projParams,
	                    GCOperatorProjector::ProjectorType projectorType,
	                    const Image* attImage,
	                    const Histogram* additiveHistogram)
	{
		std::unique_ptr<GCOperatorProjectorBase> oper;
		if (projectorType == GCOperatorProjector::SIDDON)
		{
			oper = std::make_unique<GCOperatorProjectorSiddon>(projParams);
		}
		else if (projectorType == GCOperatorProjector::DD)
		{
			oper = std::make_unique<GCOperatorProjectorDD>(projParams);
		}
		else if (projectorType == GCOperatorProjector::DD_GPU)
		{
#ifdef BUILD_CUDA
			oper = std::make_unique<GCOperatorProjectorDD_gpu>(projParams);
#else
			throw std::runtime_error("GPU projector not supported because "
			                         "Project was not compiled with CUDA ");
#endif
		}
		else
		{
			throw std::logic_error(
				"Error in forwProject: Unknown projector type (Note that the "
				"GPU Distance-Driven projector is unsupported for now)");
		}

		if (attImage != nullptr)
		{
			if constexpr (IS_FWD)
			{
				oper->setAttenuationImage(attImage);
			}
			else
			{
				oper->setAttImageForBackprojection(attImage);
			}
		}
		if (additiveHistogram != nullptr)
		{
			oper->setAddHisto(additiveHistogram);
		}

		if constexpr (IS_FWD)
		{
			std::cout << "Forward projecting all LORs ..." << std::endl;
			oper->applyA(img, projData);
			std::cout << "Done forward projecting all LORs" << std::endl;
		}
		else
		{
			std::cout << "Back projecting all LORs ..." << std::endl;
			oper->applyAH(projData, img);
			std::cout << "Done back projecting all LORs" << std::endl;
		}
	}

	void forwProject(const Scanner* scanner, const Image* img,
	                 ProjectionData* projData,
	                 GCOperatorProjector::ProjectorType projectorType,
	                 const Image* attImage,
	                 const Histogram* additiveHistogram)
	{
		const auto binIter = projData->getBinIter(1, 0);
		const GCOperatorProjectorParams projParams(binIter.get(), scanner);
		forwProject(img, projData, projParams, projectorType, attImage,
		            additiveHistogram);
	}

	void forwProject(const Scanner* scanner, const Image* img,
	                 ProjectionData* projData,
	                 const BinIterator& binIterator,
	                 GCOperatorProjector::ProjectorType projectorType,
	                 const Image* attImage,
	                 const Histogram* additiveHistogram)
	{
		const GCOperatorProjectorParams projParams(&binIterator, scanner);
		forwProject(img, projData, projParams, projectorType, attImage,
		            additiveHistogram);
	}

	void forwProject(const Image* img, ProjectionData* projData,
	                 const GCOperatorProjectorParams& projParams,
	                 GCOperatorProjector::ProjectorType projectorType,
	                 const Image* attImage,
	                 const Histogram* additiveHistogram)
	{
		project<true>(const_cast<Image*>(img), projData, projParams,
		              projectorType, attImage, additiveHistogram);
	}

	void backProject(const Scanner* scanner, Image* img,
	                 const ProjectionData* projData,
	                 GCOperatorProjector::ProjectorType projectorType,
	                 const Image* attImage,
	                 const Histogram* additiveHistogram)
	{
		const auto binIter = projData->getBinIter(1, 0);
		const GCOperatorProjectorParams projParams(binIter.get(), scanner);
		backProject(img, projData, projParams, projectorType, attImage,
		            additiveHistogram);
	}

	void backProject(const Scanner* scanner, Image* img,
	                 const ProjectionData* projData,
	                 const BinIterator& binIterator,
	                 GCOperatorProjector::ProjectorType projectorType,
	                 const Image* attImage,
	                 const Histogram* additiveHistogram)
	{
		const GCOperatorProjectorParams projParams(&binIterator, scanner);
		backProject(img, projData, projParams, projectorType, attImage,
		            additiveHistogram);
	}

	void backProject(Image* img, const ProjectionData* projData,
	                 const GCOperatorProjectorParams& projParams,
	                 GCOperatorProjector::ProjectorType projectorType,
	                 const Image* attImage,
	                 const Histogram* additiveHistogram)
	{
		project<false>(img, const_cast<ProjectionData*>(projData), projParams,
		               projectorType, attImage, additiveHistogram);
	}

} // namespace Util