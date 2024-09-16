/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "recon/GCOSEM.hpp"

#include <memory>

class GCListModeLUTOwned;
class GCHistogram3D;
class IListMode;

namespace Util
{
	void histogram3DToListModeLUT(const GCHistogram3D* histo,
	                              GCListModeLUTOwned* listMode,
	                              size_t numEvents = 0);

	template<bool RequiresAtomic>
	void convertToHistogram3D(const IProjectionData& dat,
	                          GCHistogram3D& histoOut);

	// Helper function to unify LOR loading
	GCOperatorProjectorBase::ProjectionProperties
	    getProjectionProperties(const GCScanner& scanner,
	                            const IProjectionData& dat, bin_t bin);

	GCStraightLineParam getNativeLOR(const GCScanner& scanner,
	                                 const IProjectionData& dat, bin_t binId);

	std::unique_ptr<GCOSEM> createOSEM(const GCScanner* scanner,
	                                   bool useGPU = false);


	std::tuple<GCStraightLineParam, GCVector, GCVector>
	    generateTORRandomDOI(const GCScanner* scanner, det_id_t d1, det_id_t d2,
	                         int vmax = 256);

	// Forward projection
	void forwProject(const GCScanner* scanner, const GCImage* img,
	                 IProjectionData* projData,
	                 GCOperatorProjector::ProjectorType projectorType =
	                     GCOperatorProjector::SIDDON,
	                 const GCImage* attImage = nullptr,
	                 const IHistogram* additiveHistogram = nullptr);
	void forwProject(const GCScanner* scanner, const GCImage* img,
	                 IProjectionData* projData,
	                 const GCBinIterator& binIterator,
	                 GCOperatorProjector::ProjectorType projectorType =
	                     GCOperatorProjector::SIDDON,
	                 const GCImage* attImage = nullptr,
	                 const IHistogram* additiveHistogram = nullptr);
	void forwProject(const GCImage* img, IProjectionData* projData,
	                 const GCOperatorProjectorParams& projParams,
	                 GCOperatorProjector::ProjectorType projectorType =
	                     GCOperatorProjector::SIDDON,
	                 const GCImage* attImage = nullptr,
	                 const IHistogram* additiveHistogram = nullptr);

	// Back projection
	void backProject(const GCScanner* scanner, GCImage* img,
	                 const IProjectionData* projData,
	                 GCOperatorProjector::ProjectorType projectorType =
	                     GCOperatorProjector::SIDDON,
	                 const GCImage* attImage = nullptr,
	                 const IHistogram* additiveHistogram = nullptr);
	void backProject(const GCScanner* scanner, GCImage* img,
	                 const IProjectionData* projData,
	                 const GCBinIterator& binIterator,
	                 GCOperatorProjector::ProjectorType projectorType =
	                     GCOperatorProjector::SIDDON,
	                 const GCImage* attImage = nullptr,
	                 const IHistogram* additiveHistogram = nullptr);
	void backProject(GCImage* img, const IProjectionData* projData,
	                 const GCOperatorProjectorParams& projParams,
	                 GCOperatorProjector::ProjectorType projectorType =
	                     GCOperatorProjector::SIDDON,
	                 const GCImage* attImage = nullptr,
	                 const IHistogram* additiveHistogram = nullptr);

}  // namespace Util
