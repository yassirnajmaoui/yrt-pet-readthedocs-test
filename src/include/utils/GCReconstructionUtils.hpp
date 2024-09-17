/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "recon/GCOSEM.hpp"

#include <memory>

class ListModeLUTOwned;
class Histogram3D;
class ListMode;

namespace Util
{
	void histogram3DToListModeLUT(const Histogram3D* histo,
	                              ListModeLUTOwned* listMode,
	                              size_t numEvents = 0);

	template<bool RequiresAtomic>
	void convertToHistogram3D(const ProjectionData& dat,
	                          Histogram3D& histoOut);

	// Helper function to unify LOR loading
	GCOperatorProjectorBase::ProjectionProperties
	    getProjectionProperties(const Scanner& scanner,
	                            const ProjectionData& dat, bin_t bin);

	StraightLineParam getNativeLOR(const Scanner& scanner,
	                                 const ProjectionData& dat, bin_t binId);

	std::unique_ptr<GCOSEM> createOSEM(const Scanner* scanner,
	                                   bool useGPU = false);


	std::tuple<StraightLineParam, Vector3D, Vector3D>
	    generateTORRandomDOI(const Scanner* scanner, det_id_t d1, det_id_t d2,
	                         int vmax = 256);

	// Forward projection
	void forwProject(const Scanner* scanner, const Image* img,
	                 ProjectionData* projData,
	                 GCOperatorProjector::ProjectorType projectorType =
	                     GCOperatorProjector::SIDDON,
	                 const Image* attImage = nullptr,
	                 const Histogram* additiveHistogram = nullptr);
	void forwProject(const Scanner* scanner, const Image* img,
	                 ProjectionData* projData,
	                 const BinIterator& binIterator,
	                 GCOperatorProjector::ProjectorType projectorType =
	                     GCOperatorProjector::SIDDON,
	                 const Image* attImage = nullptr,
	                 const Histogram* additiveHistogram = nullptr);
	void forwProject(const Image* img, ProjectionData* projData,
	                 const GCOperatorProjectorParams& projParams,
	                 GCOperatorProjector::ProjectorType projectorType =
	                     GCOperatorProjector::SIDDON,
	                 const Image* attImage = nullptr,
	                 const Histogram* additiveHistogram = nullptr);

	// Back projection
	void backProject(const Scanner* scanner, Image* img,
	                 const ProjectionData* projData,
	                 GCOperatorProjector::ProjectorType projectorType =
	                     GCOperatorProjector::SIDDON,
	                 const Image* attImage = nullptr,
	                 const Histogram* additiveHistogram = nullptr);
	void backProject(const Scanner* scanner, Image* img,
	                 const ProjectionData* projData,
	                 const BinIterator& binIterator,
	                 GCOperatorProjector::ProjectorType projectorType =
	                     GCOperatorProjector::SIDDON,
	                 const Image* attImage = nullptr,
	                 const Histogram* additiveHistogram = nullptr);
	void backProject(Image* img, const ProjectionData* projData,
	                 const GCOperatorProjectorParams& projParams,
	                 GCOperatorProjector::ProjectorType projectorType =
	                     GCOperatorProjector::SIDDON,
	                 const Image* attImage = nullptr,
	                 const Histogram* additiveHistogram = nullptr);

}  // namespace Util
