/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "recon/GCOSEM.hpp"

class GCOSEM_cpu : public GCOSEM
{
public:
	GCOSEM_cpu(const GCScanner* p_scanner);
	~GCOSEM_cpu() override;

protected:
	// Sens Image generator driver
	void SetupOperatorsForSensImgGen() override;
	void allocateForSensImgGen() override;
	std::unique_ptr<Image> GetLatestSensitivityImage(bool isLastSubset) override;
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

	// Common methods
	void LoadBatch(int batchId, bool forRecon) override;
	void LoadSubset(int subsetId, bool forRecon) override;

private:
	// For sensitivity image generation
	std::unique_ptr<Image> mp_tempSensImageBuffer;
	// For reconstruction
	std::unique_ptr<Image> mp_mlemImageTmp;
	std::unique_ptr<ProjectionData> mp_datTmp;

	int m_current_OSEM_subset;
};
