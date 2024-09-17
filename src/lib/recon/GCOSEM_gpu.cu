/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "recon/GCOSEM_gpu.cuh"

#include "datastruct/image/ImageDevice.cuh"
#include "datastruct/projection/ProjectionDataDevice.cuh"
#include "datastruct/projection/ProjectionSpaceKernels.cuh"
#include "operators/GCOperatorProjectorDD_gpu.cuh"
#include "utils/GCAssert.hpp"

GCOSEM_gpu::GCOSEM_gpu(const GCScanner* p_scanner)
    : GCOSEM(p_scanner),
      mpd_sensImageBuffer(nullptr),
      mpd_tempSensDataInput(nullptr),
      mpd_mlemImage(nullptr),
      mpd_mlemImageTmp(nullptr),
      mpd_dat(nullptr),
      mpd_datTmp(nullptr),
      m_current_OSEM_subset(-1)
{
	std::cout << "Creating an instance of OSEM gpu" << std::endl;

	// Since the only available projector in GPU right now is DD_GPU:
	projectorType = GCOperatorProjector::DD_GPU;
}

GCOSEM_gpu::~GCOSEM_gpu() = default;

void GCOSEM_gpu::SetupOperatorsForSensImgGen()
{
	ASSERT_MSG(projectorType == GCOperatorProjector::ProjectorType::DD_GPU,
	           "No viable projector provided");

	getBinIterators().clear();
	getBinIterators().reserve(num_OSEM_subsets);

	for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		// Create and add Bin Iterator
		getBinIterators().push_back(
		    getSensDataInput()->getBinIter(num_OSEM_subsets, subsetId));

		// Create ProjectorParams object
	}
	GCOperatorProjectorParams projParams(
	    nullptr /* Will be set later at each subset loading */, scanner, 0.f, 0,
	    flagProjPSF ? projSpacePsf_fname : "", numRays);

	mp_projector = std::make_unique<GCOperatorProjectorDD_gpu>(
	    projParams, getMainStream(), getAuxStream());

	if (attenuationImageForBackprojection != nullptr)
	{
		mp_projector->setAttImageForBackprojection(
		    attenuationImageForBackprojection);
	}
}

void GCOSEM_gpu::allocateForSensImgGen()
{
	// Allocate for image space
	mpd_sensImageBuffer =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getAuxStream());
	mpd_sensImageBuffer->allocate(true);

	// Allocate for projection space
	auto tempSensDataInput = std::make_unique<ProjectionDataDeviceOwned>(
	    scanner, getSensDataInput(), num_OSEM_subsets);
	mpd_tempSensDataInput = std::move(tempSensDataInput);
}

std::unique_ptr<Image>
    GCOSEM_gpu::GetLatestSensitivityImage(bool isLastSubset)
{
	(void)isLastSubset;  // Copy flag is obsolete since the data is not yet on
	                     // Host-side
	auto img = std::make_unique<ImageOwned>(getImageParams());
	img->allocate();
	mpd_sensImageBuffer->transferToHostMemory(img.get(), true);
	return img;
}

void GCOSEM_gpu::EndSensImgGen()
{
	// Clear temporary buffers
	mpd_sensImageBuffer = nullptr;
	mpd_tempSensDataInput = nullptr;
}

void GCOSEM_gpu::SetupOperatorsForRecon()
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

	mp_projector = std::make_unique<GCOperatorProjectorDD_gpu>(
	    projParams, getMainStream(), getAuxStream());
	if (attenuationImage != nullptr)
	{
		mp_projector->setAttenuationImage(attenuationImage);
	}
	if (addHis != nullptr)
	{
		mp_projector->setAddHisto(addHis);
	}
}

void GCOSEM_gpu::allocateForRecon()
{
	// Allocate image-space buffers
	mpd_mlemImage =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getAuxStream());
	mpd_mlemImageTmp =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getAuxStream());
	mpd_sensImageBuffer =
	    std::make_unique<ImageDeviceOwned>(getImageParams(), getAuxStream());
	mpd_mlemImage->allocate(false);
	mpd_mlemImageTmp->allocate(false);
	mpd_sensImageBuffer->allocate(false);

	// Initialize the MLEM image values to non zero
	mpd_mlemImage->setValue(INITIAL_VALUE_MLEM);

	// Initialize device's sensitivity image with the host's
	if (usingListModeInput)
	{
		mpd_sensImageBuffer->transferToDeviceMemory(getSensitivityImage(0),
		                                            true);
	}

	// Allocate projection-space buffers

	// Use the already-computed BinIterators instead of recomputing them
	std::vector<const BinIterator*> binIteratorPtrList;
	for (const auto& subsetBinIter : getBinIterators())
		binIteratorPtrList.push_back(subsetBinIter.get());

	auto dat = std::make_unique<ProjectionDataDeviceOwned>(
	    scanner, getDataInput(), binIteratorPtrList, 0.4f);
	auto datTmp = std::make_unique<ProjectionDataDeviceOwned>(dat.get());

	mpd_dat = std::move(dat);
	mpd_datTmp = std::move(datTmp);
}

void GCOSEM_gpu::EndRecon()
{
	// Transfer MLEM image Device to host
	mpd_mlemImage->transferToHostMemory(outImage, true);

	// Clear temporary buffers
	mpd_mlemImage = nullptr;
	mpd_mlemImageTmp = nullptr;
	mpd_dat = nullptr;
	mpd_datTmp = nullptr;
}

ImageBase* GCOSEM_gpu::GetSensImageBuffer()
{
	return mpd_sensImageBuffer.get();
}

ProjectionData* GCOSEM_gpu::GetSensDataInputBuffer()
{
	return mpd_tempSensDataInput.get();
}

ImageBase* GCOSEM_gpu::GetMLEMImageBuffer()
{
	return mpd_mlemImage.get();
}

ImageBase* GCOSEM_gpu::GetMLEMImageTmpBuffer()
{
	return mpd_mlemImageTmp.get();
}

ProjectionData* GCOSEM_gpu::GetMLEMDataBuffer()
{
	return mpd_dat.get();
}

ProjectionData* GCOSEM_gpu::GetMLEMDataTmpBuffer()
{
	return mpd_datTmp.get();
}

int GCOSEM_gpu::GetNumBatches(int subsetId, bool forRecon) const
{
	if (forRecon)
	{
		return mpd_dat->getNumBatches(subsetId);
	}
	return mpd_tempSensDataInput->getNumBatches(subsetId);
}

void GCOSEM_gpu::LoadBatch(int batchId, bool forRecon)
{
	std::cout << "Loading batch " << batchId + 1 << "/"
	          << GetNumBatches(m_current_OSEM_subset, forRecon) << "..."
	          << std::endl;
	if (forRecon)
	{
		mpd_dat->loadEventLORs(m_current_OSEM_subset, batchId, imageParams,
		                       getAuxStream());
		mpd_dat->allocateForProjValues(getAuxStream());
		mpd_dat->loadProjValuesFromReference(getAuxStream());

		mpd_datTmp->allocateForProjValues(getAuxStream());
	}
	else
	{
		mpd_tempSensDataInput->loadEventLORs(m_current_OSEM_subset, batchId,
		                                     imageParams, getAuxStream());
		mpd_tempSensDataInput->allocateForProjValues(getAuxStream());
		mpd_tempSensDataInput->loadProjValuesFromReference(getAuxStream());
	}
	std::cout << "Batch " << batchId + 1 << " loaded." << std::endl;
}

void GCOSEM_gpu::LoadSubset(int subsetId, bool forRecon)
{
	m_current_OSEM_subset = subsetId;

	if (forRecon && !usingListModeInput)
	{
		std::cout << "Loading OSEM subset " << subsetId + 1 << "..."
		          << std::endl;
		// Loading the right sensitivity image to the device
		mpd_sensImageBuffer->transferToDeviceMemory(
		    getSensitivityImage(m_current_OSEM_subset), true);
		std::cout << "OSEM subset loaded." << std::endl;
	}
}

void GCOSEM_gpu::CompleteMLEMIteration() {}

const cudaStream_t* GCOSEM_gpu::getAuxStream() const
{
	// TODO: Add parallel loading
	// return &m_auxStream.getStream();
	return &m_mainStream.getStream();
}

const cudaStream_t* GCOSEM_gpu::getMainStream() const
{
	return &m_mainStream.getStream();
}
