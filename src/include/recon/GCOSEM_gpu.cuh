/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/Image.hpp"
#include "datastruct/image/ImageDevice.cuh"
#include "datastruct/projection/GCProjectionDataDevice.cuh"
#include "datastruct/projection/IProjectionData.hpp"
#include "recon/GCOSEM.hpp"
#include "utils/GCGPUStream.cuh"

class GCOSEM_gpu : public GCOSEM
{
public:
	GCOSEM_gpu(const GCScanner* p_scanner);
	~GCOSEM_gpu() override;

	// Sens Image generator driver
	void SetupOperatorsForSensImgGen() override;
	void allocateForSensImgGen() override;
	std::unique_ptr<Image>
	    GetLatestSensitivityImage(bool isLastSubset) override;
	void EndSensImgGen() override;

	// Reconstruction driver
	void SetupOperatorsForRecon() override;
	void allocateForRecon() override;
	void EndRecon() override;
	void CompleteMLEMIteration() override;

	// Internal getters
	ImageBase* GetSensImageBuffer() override;
	IProjectionData* GetSensDataInputBuffer() override;
	ImageBase* GetMLEMImageBuffer() override;
	ImageBase* GetMLEMImageTmpBuffer() override;
	IProjectionData* GetMLEMDataBuffer() override;
	IProjectionData* GetMLEMDataTmpBuffer() override;
	int GetNumBatches(int subsetId, bool forRecon) const override;

	// Common methods
	void LoadBatch(int batchId, bool forRecon) override;
	void LoadSubset(int subsetId, bool forRecon) override;

private:
	const cudaStream_t* getAuxStream() const;
	const cudaStream_t* getMainStream() const;

	std::unique_ptr<ImageDeviceOwned> mpd_sensImageBuffer;
	std::unique_ptr<GCProjectionDataDeviceOwned> mpd_tempSensDataInput;
	std::unique_ptr<ImageDeviceOwned> mpd_mlemImage;
	std::unique_ptr<ImageDeviceOwned> mpd_mlemImageTmp;
	std::unique_ptr<GCProjectionDataDeviceOwned> mpd_dat;
	std::unique_ptr<GCProjectionDataDeviceOwned> mpd_datTmp;

	int m_current_OSEM_subset;

	GCGPUStream m_mainStream;
	// GCGPUStream m_auxStream;

	// TODO: Potential optimisation: Avoid transferring the Scanner LUT twice
	//  (once for gensensimg and another for recon)
};
