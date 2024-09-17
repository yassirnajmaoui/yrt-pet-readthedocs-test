/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/Image.hpp"
#include "datastruct/image/ImageDevice.cuh"
#include "datastruct/projection/ProjectionDataDevice.cuh"
#include "datastruct/projection/ProjectionData.hpp"
#include "recon/OSEM.hpp"
#include "utils/GPUStream.cuh"

class OSEM_gpu : public OSEM
{
public:
	OSEM_gpu(const Scanner* p_scanner);
	~OSEM_gpu() override;

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
	ProjectionData* GetSensDataInputBuffer() override;
	ImageBase* GetMLEMImageBuffer() override;
	ImageBase* GetMLEMImageTmpBuffer() override;
	ProjectionData* GetMLEMDataBuffer() override;
	ProjectionData* GetMLEMDataTmpBuffer() override;
	int GetNumBatches(int subsetId, bool forRecon) const override;

	// Common methods
	void LoadBatch(int batchId, bool forRecon) override;
	void LoadSubset(int subsetId, bool forRecon) override;

private:
	const cudaStream_t* getAuxStream() const;
	const cudaStream_t* getMainStream() const;

	std::unique_ptr<ImageDeviceOwned> mpd_sensImageBuffer;
	std::unique_ptr<ProjectionDataDeviceOwned> mpd_tempSensDataInput;
	std::unique_ptr<ImageDeviceOwned> mpd_mlemImage;
	std::unique_ptr<ImageDeviceOwned> mpd_mlemImageTmp;
	std::unique_ptr<ProjectionDataDeviceOwned> mpd_dat;
	std::unique_ptr<ProjectionDataDeviceOwned> mpd_datTmp;

	int m_current_OSEM_subset;

	GPUStream m_mainStream;
	// GPUStream m_auxStream;

	// TODO: Potential optimisation: Avoid transferring the Scanner LUT twice
	//  (once for gensensimg and another for recon)
};
