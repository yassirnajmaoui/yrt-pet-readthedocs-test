/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "operators/GCOperatorDevice.cuh"
#include "operators/GCOperatorProjector.hpp"


class GCProjectionDataDevice;
class ImageDevice;

class GCOperatorProjectorDD_gpu : public GCOperatorProjectorDevice
{
public:
	GCOperatorProjectorDD_gpu(const GCOperatorProjectorParams& projParams,
	                          bool p_synchronized = true,
	                          const cudaStream_t* mainStream = nullptr,
	                          const cudaStream_t* auxStream = nullptr);

	void applyA(const GCVariable* in, GCVariable* out) override;
	void applyAH(const GCVariable* in, GCVariable* out) override;

private:
	void applyAttenuationOnLoadedBatchIfNeeded(
	    const GCProjectionDataDevice* imgProjData, bool duringForward);
	void applyAttenuationOnLoadedBatchIfNeeded(
	    const GCProjectionDataDevice* imgProjData,
	    GCProjectionDataDevice* destProjData, bool duringForward);
	void
	    applyAdditiveOnLoadedBatchIfNeeded(GCProjectionDataDevice* imgProjData);

	void applyAttenuationFactors(const GCProjectionDataDevice* attImgProj,
	                             const GCProjectionDataDevice* imgProj,
	                             GCProjectionDataDevice* destProj,
	                             float unitFactor);

	template <bool IsForward>
	void applyOnLoadedBatch(GCProjectionDataDevice* dat, ImageDevice* img);

	template <bool IsForward, bool HasTOF>
	static void launchKernel(float* pd_projValues, float* pd_image,
	                         const float4* pd_lorDet1Pos,
	                         const float4* pd_lorDet2Pos,
	                         const float4* pd_lorDet1Orient,
	                         const float4* pd_lorDet2Orient,
	                         const float* pd_lorTOFValue,
	                         const GCTimeOfFlightHelper* pd_tofHelper,
	                         GCCUScannerParams scannerParams,
	                         GCCUImageParams imgParams, size_t batchSize,
	                         unsigned int gridSize, unsigned int blockSize,
	                         const cudaStream_t* stream, bool synchronize);
};
