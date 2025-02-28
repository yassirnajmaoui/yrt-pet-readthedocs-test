/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/ImageDevice.cuh"
#include "datastruct/projection/ProjectionDataDevice.cuh"
#include "operators/DeviceSynchronized.cuh"
#include "operators/OperatorProjectorBase.hpp"
#include "operators/ProjectionPsfManagerDevice.cuh"
#include "operators/TimeOfFlight.hpp"
#include "utils/DeviceObject.cuh"
#include "utils/GPUTypes.cuh"


class OperatorProjectorDevice : public OperatorProjectorBase,
                                public DeviceSynchronized
{
public:
	OperatorProjectorDevice() = delete;

	size_t getBatchSize() const;
	void setupProjPsfManager(const std::string& psfFilename);

	unsigned int getGridSize() const;
	unsigned int getBlockSize() const;

	bool requiresIntermediaryProjData() const;
	void setupTOFHelper(float tofWidth_ps, int tofNumStd = -1);

	void applyA(const Variable* in, Variable* out) override;
	void applyAH(const Variable* in, Variable* out) override;

	void applyA(const Variable* in, Variable* out, bool synchronize);
	void applyAH(const Variable* in, Variable* out, bool synchronize);

protected:
	explicit OperatorProjectorDevice(
	    const OperatorProjectorParams& pr_projParams,
	    const cudaStream_t* pp_mainStream = nullptr,
	    const cudaStream_t* pp_auxStream = nullptr);

	virtual void applyAOnLoadedBatch(ImageDevice& img,
	                                 ProjectionDataDevice& dat,
	                                 bool synchronize) = 0;
	virtual void applyAHOnLoadedBatch(ProjectionDataDevice& dat,
	                                  ImageDevice& img, bool synchronize) = 0;

	void setBatchSize(size_t newBatchSize);

	const TimeOfFlightHelper* getTOFHelperDevicePointer() const;
	const float* getProjPsfKernelsDevicePointer(bool flipped) const;

	// Projection-domain PSF
	std::unique_ptr<ProjectionPsfManagerDevice> mp_projPsfManager;

private:
	size_t m_batchSize;
	GPULaunchParams m_launchParams{};

	// Time of flight
	std::unique_ptr<DeviceObject<TimeOfFlightHelper>> mp_tofHelper;

	// For attenuation correction
	std::unique_ptr<ImageDeviceOwned> mp_attImageDevice;
	// For attenuation correction
	std::unique_ptr<ImageDeviceOwned> mp_attImageForBackprojectionDevice;
	// For Attenuation correction or Additive correction
	std::unique_ptr<ProjectionDataDeviceOwned> mp_intermediaryProjData;
};
