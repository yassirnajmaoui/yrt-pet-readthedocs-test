/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/ImageBase.hpp"
#include "datastruct/image/ImageDevice.cuh"
#include "datastruct/projection/GCProjectionDataDevice.cuh"
#include "operators/GCOperator.hpp"
#include "operators/GCOperatorProjector.hpp"
#include "recon/GCCUParameters.hpp"
#include "utils/GCDeviceObject.cuh"
#include "utils/GCGPUTypes.cuh"

#include <cuda_runtime_api.h>

namespace Util
{
	// Takes a reference of an image, and automatically determines
	// the 3D grid/block parameters.
	GCGPULaunchParams3D initiateDeviceParameters(const ImageParams& params);
	// Takes a size of the data to be processed, and automatically determines
	// the grid/block parameters
	GCGPULaunchParams initiateDeviceParameters(size_t batchSize);
}  // namespace Util

class GCOperatorDevice : public GCOperator
{
	// Nothing here for now...

public:
	static GCCUScannerParams getCUScannerParams(const GCScanner& scanner);
	static GCCUImageParams getCUImageParams(const ImageParams& imgParams);

protected:
	GCOperatorDevice() = default;
};

class GCOperatorProjectorDevice : public GCOperatorProjectorBase,
                                  public GCOperatorDevice
{
public:
	GCOperatorProjectorDevice() = delete;

	size_t getBatchSize() const;

	unsigned int getGridSize() const;
	unsigned int getBlockSize() const;
	bool isSynchronized() const;

	bool requiresIntermediaryProjData() const;

	void setAttImage(const Image* attImage) override;
	void setAttImageForBackprojection(const Image* attImage) override;
	void setAddHisto(const IHistogram *p_addHisto) override;
	void setupTOFHelper(float tofWidth_ps, int tofNumStd = -1);

protected:
	GCOperatorProjectorDevice(const GCOperatorProjectorParams& projParams,
	                          bool p_synchronized = true,
	                          const cudaStream_t* pp_mainStream = nullptr,
	                          const cudaStream_t* pp_auxStream = nullptr);

	const cudaStream_t* getMainStream() const;
	const cudaStream_t* getAuxStream() const;

	void setBatchSize(size_t newBatchSize);

	GCProjectionDataDeviceOwned& getIntermediaryProjData();
	const ImageDevice& getAttImageDevice() const;
	const ImageDevice& getAttImageForBackprojectionDevice() const;
	void prepareIntermediaryBuffer(const GCProjectionDataDevice* orig);
	void prepareIntermediaryBufferIfNeeded(const GCProjectionDataDevice* orig);

	const GCTimeOfFlightHelper* getTOFHelperDevicePointer() const;

private:
	size_t m_batchSize;
	GCGPULaunchParams m_launchParams{};
	const cudaStream_t* mp_mainStream;
	const cudaStream_t* mp_auxStream;
	bool m_synchonized;

	std::unique_ptr<GCDeviceObject<GCTimeOfFlightHelper>> mp_tofHelper;

	// For attenuation correction
	std::unique_ptr<ImageDeviceOwned> mp_attImageDevice;
	// For attenuation correction
	std::unique_ptr<ImageDeviceOwned> mp_attImageForBackprojectionDevice;
	// For Attenuation correction or Additive correction
	std::unique_ptr<GCProjectionDataDeviceOwned> mp_intermediaryProjData;
};
