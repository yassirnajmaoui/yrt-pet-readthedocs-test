/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/ImageBase.hpp"
#include "datastruct/projection/GCLORsDevice.cuh"
#include "datastruct/projection/GCProjectionList.hpp"
#include "datastruct/projection/IProjectionData.hpp"
#include "datastruct/scanner/GCScannerDevice.cuh"
#include "utils/GCDeviceArray.cuh"
#include "utils/GCGPUTypes.cuh"
#include "utils/GCPageLockedBuffer.cuh"

#include <memory>

class IHistogram;

class GCProjectionDataDevice : public GCProjectionList
{
public:
	// The Scanner LUT has to be loaded to device, but the BinIterators have
	// already been generated
	GCProjectionDataDevice(const GCScanner* pp_scanner,
	                       const IProjectionData* pp_reference,
	                       std::vector<const BinIterator*> pp_binIteratorList,
	                       float shareOfMemoryToUse = DefaultMemoryShare);
	// The Scanner LUT has to be loaded to device AND the BinIterators have to
	// be generated
	GCProjectionDataDevice(const GCScanner* pp_scanner,
	                       const IProjectionData* pp_reference,
	                       int num_OSEM_subsets = 1,
	                       float shareOfMemoryToUse = DefaultMemoryShare);
	// The Scanner LUT is already in the device, but still need to generate the
	// BinIterators
	GCProjectionDataDevice(std::shared_ptr<GCScannerDevice> pp_scannerDevice,
	                       const IProjectionData* pp_reference,
	                       int num_OSEM_subsets = 1,
	                       float shareOfMemoryToUse = DefaultMemoryShare);
	// The Scanner LUT AND the lines of responses are already on device, but the
	// BinIterators have already been generated
	GCProjectionDataDevice(std::shared_ptr<GCLORsDevice> pp_LORs,
	                       const IProjectionData* pp_reference,
	                       int num_OSEM_subsets = 1,
	                       float shareOfMemoryToUse = DefaultMemoryShare);
	// The Scanner LUT AND the lines of responses are already on the device, and
	// no need to generate the BinIterators
	GCProjectionDataDevice(std::shared_ptr<GCLORsDevice> pp_LORs,
	                       const IProjectionData* pp_reference,
	                       std::vector<const BinIterator*> pp_binIteratorList,
	                       float shareOfMemoryToUse = DefaultMemoryShare);
	// Proxy for the above
	GCProjectionDataDevice(const GCProjectionDataDevice* orig);

	// Load the events' detector ids from a specific subset&batch id and prepare
	// the projection values buffer
	void loadEventLORs(size_t subsetId, size_t batchId,
	                   const ImageParams& imgParams,
	                   const cudaStream_t* stream = nullptr);
	void loadProjValuesFromReference(const cudaStream_t* stream = nullptr);
	void loadProjValuesFromHost(const IProjectionData* src,
	                            const cudaStream_t* stream);
	void loadProjValuesFromHost(const IProjectionData* src,
	                            const IHistogram* histo,
	                            const cudaStream_t* stream);
	void transferProjValuesToHost(IProjectionData* projDataDest,
	                              const cudaStream_t* stream = nullptr) const;

	std::shared_ptr<GCScannerDevice> getScannerDevice() const;
	const GCScanner* getScanner() const;

	// Gets the size of the last-loaded batch
	size_t getCurrentBatchSize() const;
	// Gets the index of the last-loaded batch
	size_t getCurrentBatchId() const;
	// Get the index of the last-loaded subset
	size_t getCurrentSubsetId() const;

	virtual float* getProjValuesDevicePointer() = 0;
	virtual const float* getProjValuesDevicePointer() const = 0;
	const float4* getLorDet1PosDevicePointer() const;
	const float4* getLorDet1OrientDevicePointer() const;
	const float4* getLorDet2PosDevicePointer() const;
	const float4* getLorDet2OrientDevicePointer() const;
	const float* getLorTOFValueDevicePointer() const;

	float getProjectionValue(bin_t id) const override;
	void setProjectionValue(bin_t id, float val) override;
	void clearProjections(float value) override;
	void clearProjections(float value, const cudaStream_t* stream);
	void clearProjections(const cudaStream_t* stream = nullptr);
	void divideMeasurements(const IProjectionData* measurements,
	                        const BinIterator* binIter) override;
	void divideMeasurements(const IProjectionData* measurements,
	                        const BinIterator* binIter,
	                        const cudaStream_t* stream);
	void addProjValues(const GCProjectionDataDevice* projValues,
	                   const cudaStream_t* stream);
	const GCGPUBatchSetup& getBatchSetup(size_t subsetId) const;
	size_t getNumBatches(size_t subsetId) const;
	bool areLORsGathered() const;

	static constexpr size_t MemoryUsagePerEvent =
	    sizeof(float) + GCLORsDevice::MemoryUsagePerLOR;

	// Use 90% of what is available
	static constexpr float DefaultMemoryShare = 0.9f;

protected:
	virtual void loadProjValuesFromHostInternal(const IProjectionData* src,
	                                            const IHistogram* histo,
	                                            const cudaStream_t* stream);

	// For Host->Device data transfers
	mutable GCPageLockedBuffer<float> m_tempBuffer;

	// We need all the BinIterators in order to be able to properly load the
	// data from Host to device (and vice-verse)
	std::vector<const BinIterator*> mp_binIteratorList;

private:
	void createBinIterators(int num_OSEM_subsets);
	void createBatchSetups(float shareOfMemoryToUse);

	const GCScanner* mp_scanner;

	std::shared_ptr<GCLORsDevice> mp_LORs;
	std::vector<GCGPUBatchSetup> m_batchSetups;  // One batch setup per subset

	// In case we need to compute our own BinIterators
	std::vector<std::unique_ptr<BinIterator>> m_binIterators;
};

class GCProjectionDataDeviceOwned : public GCProjectionDataDevice
{
public:
	GCProjectionDataDeviceOwned(
	    const GCScanner* pp_scanner, const IProjectionData* pp_reference,
	    std::vector<const BinIterator*> pp_binIteratorList,
	    float shareOfMemoryToUse = DefaultMemoryShare);
	GCProjectionDataDeviceOwned(const GCScanner* pp_scanner,
	                            const IProjectionData* pp_reference,
	                            int num_OSEM_subsets = 1,
	                            float shareOfMemoryToUse = DefaultMemoryShare);
	GCProjectionDataDeviceOwned(
	    std::shared_ptr<GCScannerDevice> pp_scannerDevice,
	    const IProjectionData* pp_reference, int num_OSEM_subsets = 1,
	    float shareOfMemoryToUse = DefaultMemoryShare);
	GCProjectionDataDeviceOwned(std::shared_ptr<GCLORsDevice> pp_LORs,
	                            const IProjectionData* pp_reference,
	                            int num_OSEM_subsets = 1,
	                            float shareOfMemoryToUse = DefaultMemoryShare);
	GCProjectionDataDeviceOwned(
	    std::shared_ptr<GCLORsDevice> pp_LORs,
	    const IProjectionData* pp_reference,
	    std::vector<const BinIterator*> pp_binIteratorList,
	    float shareOfMemoryToUse = DefaultMemoryShare);
	GCProjectionDataDeviceOwned(const GCProjectionDataDevice* orig);

	~GCProjectionDataDeviceOwned() override = default;

	void allocateForProjValues(const cudaStream_t* stream = nullptr);

	float* getProjValuesDevicePointer() override;
	const float* getProjValuesDevicePointer() const override;

protected:
	void loadProjValuesFromHostInternal(const IProjectionData* src,
	                                    const IHistogram* histo,
	                                    const cudaStream_t* stream) override;

private:
	std::unique_ptr<GCDeviceArray<float>> mp_projValues;
};

class GCProjectionDataDeviceAlias : public GCProjectionDataDevice
{
public:
	GCProjectionDataDeviceAlias(
	    const GCScanner* pp_scanner, const IProjectionData* pp_reference,
	    std::vector<const BinIterator*> pp_binIteratorList,
	    float shareOfMemoryToUse = DefaultMemoryShare);
	GCProjectionDataDeviceAlias(const GCScanner* pp_scanner,
	                            const IProjectionData* pp_reference,
	                            int num_OSEM_subsets = 1,
	                            float shareOfMemoryToUse = DefaultMemoryShare);
	GCProjectionDataDeviceAlias(
	    std::shared_ptr<GCScannerDevice> pp_scannerDevice,
	    const IProjectionData* pp_reference, int num_OSEM_subsets = 1,
	    float shareOfMemoryToUse = DefaultMemoryShare);
	GCProjectionDataDeviceAlias(std::shared_ptr<GCLORsDevice> pp_LORs,
	                            const IProjectionData* pp_reference,
	                            int num_OSEM_subsets = 1,
	                            float shareOfMemoryToUse = DefaultMemoryShare);
	GCProjectionDataDeviceAlias(
	    std::shared_ptr<GCLORsDevice> pp_LORs,
	    const IProjectionData* pp_reference,
	    std::vector<const BinIterator*> pp_binIteratorList,
	    float shareOfMemoryToUse = DefaultMemoryShare);
	GCProjectionDataDeviceAlias(const GCProjectionDataDevice* orig);

	float* getProjValuesDevicePointer() override;
	const float* getProjValuesDevicePointer() const override;
	size_t getProjValuesDevicePointerInULL() const;

	void setProjValuesDevicePointer(float* ppd_devicePointer);
	void setProjValuesDevicePointer(size_t ppd_pointerInULL);
	bool isDevicePointerSet() const;

private:
	float* mpd_devicePointer;
};
