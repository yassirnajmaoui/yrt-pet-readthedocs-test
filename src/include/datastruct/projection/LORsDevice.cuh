/*
* This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/projection/BinIterator.hpp"
#include "datastruct/scanner/ScannerDevice.cuh"
#include "utils/GCGPUTypes.cuh"

#include <memory>

class ProjectionData;
class Scanner;
class ImageParams;

class LORsDevice
{
public:
	LORsDevice(std::shared_ptr<ScannerDevice> pp_scannerDevice);
	LORsDevice(const Scanner* pp_scanner);

	// Load the events' detector ids from a specific subset&batch id
	void loadEventLORs(const BinIterator& binIter,
	                   const GCGPUBatchSetup& batchSetup, size_t subsetId,
	                   size_t batchId, const ProjectionData& reference,
	                   const ImageParams& imgParams,
	                   const cudaStream_t* stream = nullptr);

	std::shared_ptr<ScannerDevice> getScannerDevice() const;
	const Scanner* getScanner() const;

	// Gets the size of the last-loaded batch
	size_t getLoadedBatchSize() const;
	// Gets the index of the last-loaded batch
	size_t getLoadedBatchId() const;
	// Gets the index of the last-loaded subset
	size_t getLoadedSubsetId() const;

	const float4* getLorDet1PosDevicePointer() const;
	const float4* getLorDet1OrientDevicePointer() const;
	const float4* getLorDet2PosDevicePointer() const;
	const float4* getLorDet2OrientDevicePointer() const;
	const float* getLorTOFValueDevicePointer() const;
	float4* getLorDet1PosDevicePointer();
	float4* getLorDet1OrientDevicePointer();
	float4* getLorDet2PosDevicePointer();
	float4* getLorDet2OrientDevicePointer();
	float* getLorTOFValueDevicePointer();
	bool areLORsGathered() const;

	static constexpr size_t MemoryUsagePerLOR =
	    sizeof(float4) * 4 + sizeof(float);

private:
	void initializeDeviceArrays();
	void allocateForLORs(bool hasTOF, const cudaStream_t* stream = nullptr);

	std::shared_ptr<ScannerDevice> mp_scannerDevice;

	std::unique_ptr<GCDeviceArray<float4>> mp_lorDet1Pos;
	std::unique_ptr<GCDeviceArray<float4>> mp_lorDet2Pos;
	std::unique_ptr<GCDeviceArray<float4>> mp_lorDet1Orient;
	std::unique_ptr<GCDeviceArray<float4>> mp_lorDet2Orient;
	std::unique_ptr<GCDeviceArray<float>> mp_lorTOFValue;
	bool m_areLORsGathered;
	size_t m_loadedBatchSize;
	size_t m_loadedBatchId;
	size_t m_loadedSubsetId;
};

