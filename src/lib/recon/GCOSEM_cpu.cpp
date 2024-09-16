/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "recon/GCOSEM_cpu.hpp"

#include "datastruct/projection/GCProjectionList.hpp"
#include "datastruct/projection/IListMode.hpp"
#include "operators/GCOperatorProjectorDD.hpp"
#include "operators/GCOperatorProjectorSiddon.hpp"

#include <utility>

GCOSEM_cpu::GCOSEM_cpu(const GCScanner* p_scanner)
    : GCOSEM(p_scanner),
      mp_tempSensImageBuffer(nullptr),
      mp_mlemImageTmp(nullptr),
      mp_datTmp(nullptr),
      m_current_OSEM_subset(-1)
{
}

GCOSEM_cpu::~GCOSEM_cpu() = default;


void GCOSEM_cpu::allocateForSensImgGen()
{
	auto tempSensImageBuffer = std::make_unique<ImageOwned>(getImageParams());
	tempSensImageBuffer->allocate();
	mp_tempSensImageBuffer = std::move(tempSensImageBuffer);
}

void GCOSEM_cpu::SetupOperatorsForSensImgGen()
{
	// TODO: Unify this in GCOSEM (avoids the copy-paste)
	getBinIterators().clear();
	getBinIterators().reserve(num_OSEM_subsets);

	for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		// Create and add Bin Iterator
		getBinIterators().push_back(
		    getSensDataInput()->getBinIter(num_OSEM_subsets, subsetId));
	}

	// Create ProjectorParams object
	GCOperatorProjectorParams projParams(
	    nullptr /* Will be set later at each subset loading */, scanner, 0.f, 0,
	    flagProjPSF ? projSpacePsf_fname : "", numRays);

	if (projectorType == GCOperatorProjector::ProjectorType::SIDDON)
	{
		mp_projector = std::make_unique<GCOperatorProjectorSiddon>(projParams);
	}
	else if (projectorType == GCOperatorProjector::ProjectorType::DD)
	{
		mp_projector = std::make_unique<GCOperatorProjectorDD>(projParams);
	}

	if (attenuationImageForBackprojection != nullptr)
	{
		mp_projector->setAttImageForBackprojection(
		    attenuationImageForBackprojection);
	}
}

std::unique_ptr<Image>
    GCOSEM_cpu::GetLatestSensitivityImage(bool isLastSubset)
{
	// This will dereference mp_tempSensImageBuffer
	auto img = std::move(mp_tempSensImageBuffer);

	// Which requires another allocation for the next subset
	if (!isLastSubset)
	{
		allocateForSensImgGen();
	}

	return img;
}

void GCOSEM_cpu::EndSensImgGen()
{
	// Clear temporary buffers
	mp_tempSensImageBuffer = nullptr;
}

ImageBase* GCOSEM_cpu::GetSensImageBuffer()
{
	if (mp_tempSensImageBuffer != nullptr)
	{
		return mp_tempSensImageBuffer.get();
	}
	// In case we are not currently generating the sensitivity image
	return getSensitivityImage(usingListModeInput ? 0 : m_current_OSEM_subset);
}

IProjectionData* GCOSEM_cpu::GetSensDataInputBuffer()
{
	// Since in the CPU version, the projection data is unchanged from the
	// original and stays in the Host.
	return getSensDataInput();
}

ImageBase* GCOSEM_cpu::GetMLEMImageBuffer()
{
	return outImage;
}
ImageBase* GCOSEM_cpu::GetMLEMImageTmpBuffer()
{
	return mp_mlemImageTmp.get();
}

IProjectionData* GCOSEM_cpu::GetMLEMDataBuffer()
{
	return getDataInput();
}

IProjectionData* GCOSEM_cpu::GetMLEMDataTmpBuffer()
{
	return mp_datTmp.get();
}

void GCOSEM_cpu::SetupOperatorsForRecon()
{
	getBinIterators().clear();
	getBinIterators().reserve(num_OSEM_subsets);

	for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		getBinIterators().push_back(
		    getDataInput()->getBinIter(num_OSEM_subsets, subsetId));
	}

	// Create ProjectorParams object
	GCOperatorProjectorParams projParams(
	    nullptr /* Will be set later at each subset loading */, scanner,
	    flagProjTOF ? tofWidth_ps : 0.f, flagProjTOF ? tofNumStd : 0,
	    flagProjPSF ? projSpacePsf_fname : "", numRays);

	if (projectorType == GCOperatorProjector::SIDDON)
	{
		mp_projector = std::make_unique<GCOperatorProjectorSiddon>(projParams);
	}
	else if (projectorType == GCOperatorProjector::DD)
	{
		mp_projector = std::make_unique<GCOperatorProjectorDD>(projParams);
	}

	if (attenuationImage != nullptr)
	{
		mp_projector->setAttenuationImage(attenuationImage);
	}
	if (addHis != nullptr)
	{
		mp_projector->setAddHisto(addHis);
	}
}

void GCOSEM_cpu::allocateForRecon()
{
	// Allocate for projection-space buffers
	mp_datTmp = std::make_unique<GCProjectionListOwned>(getDataInput());
	reinterpret_cast<GCProjectionListOwned*>(mp_datTmp.get())->allocate();

	// Allocate for image-space buffers
	mp_mlemImageTmp = std::make_unique<ImageOwned>(getImageParams());
	reinterpret_cast<ImageOwned*>(mp_mlemImageTmp.get())->allocate();

	// Initialize output image
	GetMLEMImageBuffer()->setValue(INITIAL_VALUE_MLEM);

	// Apply mask image
	std::cout << "Applying threshold" << std::endl;
	auto applyMask = [this](const Image* maskImage) -> void
	{ GetMLEMImageBuffer()->applyThreshold(maskImage, 0.0, 0.0, 0.0, 0.0, 1); };
	if (maskImage != nullptr)
	{
		applyMask(maskImage);
	}
	else if (num_OSEM_subsets == 1 || usingListModeInput)
	{
		// No need to sum all sensitivity images, just use the only one
		applyMask(getSensitivityImage(0));
	}
	else
	{
		std::cout << "Summing sensitivity images to generate mask image..."
		          << std::endl;
		auto sensitivityImageSum =
		    std::make_unique<ImageOwned>(getImageParams());
		sensitivityImageSum->allocate();
		for (int i = 0; i < num_OSEM_subsets; ++i)
		{
			getSensitivityImage(i)->addFirstImageToSecond(
			    sensitivityImageSum.get());
		}
		applyMask(sensitivityImageSum.get());
		std::cout << "Done summing" << std::endl;
	}
	std::cout << "Threshold applied" << std::endl;
}

void GCOSEM_cpu::EndRecon()
{
	// Clear temporary buffers
	mp_mlemImageTmp = nullptr;
	mp_datTmp = nullptr;
}

void GCOSEM_cpu::LoadBatch(int batchId, bool forRecon)
{
	// No-op on CPU
	(void)forRecon;
	(void)batchId;
}

void GCOSEM_cpu::LoadSubset(int subsetId, bool forRecon)
{
	(void)forRecon;
	m_current_OSEM_subset = subsetId;
}

void GCOSEM_cpu::CompleteMLEMIteration() {}
