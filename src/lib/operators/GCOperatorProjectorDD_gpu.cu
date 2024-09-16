/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/GCOperatorProjectorDD_gpu.cuh"

#include "datastruct/image/Image.hpp"
#include "datastruct/image/ImageDevice.cuh"
#include "datastruct/projection/GCProjectionDataDevice.cuh"
#include "operators/GCOperatorProjectorDD_kernels.cuh"
#include "utils/GCAssert.hpp"
#include "utils/GCGPUUtils.cuh"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

void py_setup_gcoperatorprojectordd_gpu(py::module& m)
{
	auto c = py::class_<GCOperatorProjectorDD_gpu, GCOperatorProjectorDevice>(
	    m, "GCOperatorProjectorDD_gpu");
	c.def(py::init<const GCOperatorProjectorParams&>(), py::arg("projParams"));
	c.def(
	    "applyA",
	    [](GCOperatorProjector& self, const ImageDevice* img,
	       IProjectionData* proj) { self.applyA(img, proj); },
	    py::arg("img"), py::arg("proj"));
	c.def(
	    "applyA",
	    [](GCOperatorProjector& self, const Image* img, IProjectionData* proj)
	    { self.applyA(img, proj); }, py::arg("img"), py::arg("proj"));
	c.def(
	    "applyA",
	    [](GCOperatorProjector& self, const ImageDevice* img,
	       GCProjectionDataDevice* proj) { self.applyA(img, proj); },
	    py::arg("img"), py::arg("proj"));
	c.def(
	    "applyA",
	    [](GCOperatorProjector& self, const Image* img,
	       GCProjectionDataDevice* proj) { self.applyA(img, proj); },
	    py::arg("img"), py::arg("proj"));

	c.def(
	    "applyAH",
	    [](GCOperatorProjector& self, const IProjectionData* proj, Image* img)
	    { self.applyAH(proj, img); }, py::arg("proj"), py::arg("img"));
	c.def(
	    "applyAH",
	    [](GCOperatorProjector& self, const IProjectionData* proj,
	       ImageDevice* img) { self.applyAH(proj, img); },
	    py::arg("proj"), py::arg("img"));
	c.def(
	    "applyAH",
	    [](GCOperatorProjector& self, const GCProjectionDataDevice* proj,
	       Image* img) { self.applyAH(proj, img); },
	    py::arg("proj"), py::arg("img"));
	c.def(
	    "applyAH",
	    [](GCOperatorProjector& self, const GCProjectionDataDevice* proj,
	       ImageDevice* img) { self.applyAH(proj, img); },
	    py::arg("proj"), py::arg("img"));
}
#endif

GCOperatorProjectorDD_gpu::GCOperatorProjectorDD_gpu(
    const GCOperatorProjectorParams& projParams, bool p_synchronized,
    const cudaStream_t* mainStream, const cudaStream_t* auxStream)
    : GCOperatorProjectorDevice(projParams, p_synchronized, mainStream,
                                auxStream)
{
}

void GCOperatorProjectorDD_gpu::applyA(const GCVariable* in, GCVariable* out)
{
	auto* img_in_const = dynamic_cast<const ImageDevice*>(in);
	auto* dat_out = dynamic_cast<GCProjectionDataDevice*>(out);

	// In case the user provided a host-side image
	std::unique_ptr<ImageDeviceOwned> deviceImg_out = nullptr;
	ImageDevice* img_in = nullptr;
	if (img_in_const == nullptr)
	{
		const auto* hostImg_in = dynamic_cast<const Image*>(in);
		ASSERT_MSG(
		    hostImg_in != nullptr,
		    "The image provided is not a ImageDevice nor a Image (host)");

		deviceImg_out = std::make_unique<ImageDeviceOwned>(
		    hostImg_in->getParams(), getAuxStream());
		deviceImg_out->allocate(true);
		deviceImg_out->transferToDeviceMemory(hostImg_in, true);

		// Use owned ImageDevice
		img_in = deviceImg_out.get();
	}
	else
	{
		img_in = const_cast<ImageDevice*>(img_in_const);
		ASSERT_MSG(img_in != nullptr, "ImageDevice is null. Cast failed");
	}

	// In case the user provided a Host-side IProjectionData
	bool isProjDataDeviceOwned = false;
	std::unique_ptr<GCProjectionDataDeviceOwned> deviceDat_out = nullptr;
	IProjectionData* hostDat_out = nullptr;
	if (dat_out == nullptr)
	{
		hostDat_out = dynamic_cast<IProjectionData*>(out);
		ASSERT_MSG(
		    hostDat_out != nullptr,
		    "The Projection Data provded is not a GCProjectionDataDevice "
		    "nor a IProjectionData (host)");

		std::vector<const GCBinIterator*> binIterators;
		binIterators.push_back(binIter);  // We project only one subset
		deviceDat_out = std::make_unique<GCProjectionDataDeviceOwned>(
		    getScanner(), hostDat_out, binIterators);

		// Use owned GCProjectionDataDevice
		dat_out = deviceDat_out.get();
		isProjDataDeviceOwned = true;
	}

	if (!isProjDataDeviceOwned)
	{
		std::cout << "Forward projecting current batch..." << std::endl;
		applyOnLoadedBatch<true>(dat_out, img_in);
		std::cout << "Done Forward projecting current batch." << std::endl;
		applyAttenuationOnLoadedBatchIfNeeded(dat_out, dat_out, true);
		applyAdditiveOnLoadedBatchIfNeeded(dat_out);
	}
	else
	{
		// Iterate over all the batches of the current subset
		const size_t numBatches = dat_out->getBatchSetup(0).getNumBatches();
		const ImageParams& imgParams = img_in->getParams();
		for (size_t batchId = 0; batchId < numBatches; batchId++)
		{
			std::cout << "Loading batch " << batchId + 1 << "/" << numBatches
			          << "..." << std::endl;
			dat_out->loadEventLORs(0, batchId, imgParams, getAuxStream());
			deviceDat_out->allocateForProjValues(getAuxStream());
			dat_out->clearProjections(getMainStream());
			std::cout << "Batch " << batchId + 1 << " loaded." << std::endl;
			std::cout << "Forward projecting batch..." << std::endl;
			applyOnLoadedBatch<true>(dat_out, img_in);
			std::cout << "Done forward projecting batch." << std::endl;
			applyAttenuationOnLoadedBatchIfNeeded(dat_out, dat_out, true);
			applyAdditiveOnLoadedBatchIfNeeded(dat_out);
			std::cout << "Transferring batch to Host..." << std::endl;
			dat_out->transferProjValuesToHost(hostDat_out, getAuxStream());
			std::cout << "Done transferring batch to host." << std::endl;
		}
	}
}

void GCOperatorProjectorDD_gpu::applyAH(const GCVariable* in, GCVariable* out)
{
	auto* dat_in_const = dynamic_cast<const GCProjectionDataDevice*>(in);
	auto* img_out = dynamic_cast<ImageDevice*>(out);

	bool isImageDeviceOwned = false;

	// In case the user provided a host-side image
	std::unique_ptr<ImageDeviceOwned> deviceImg_out = nullptr;
	Image* hostImg_out = nullptr;
	if (img_out == nullptr)
	{
		hostImg_out = dynamic_cast<Image*>(out);
		ASSERT_MSG(
		    hostImg_out != nullptr,
		    "The image provided is not a ImageDevice nor a Image (host)");

		deviceImg_out = std::make_unique<ImageDeviceOwned>(
		    hostImg_out->getParams(), getAuxStream());
		deviceImg_out->allocate(false);
		deviceImg_out->transferToDeviceMemory(hostImg_out, false);

		// Use owned ImageDevice
		img_out = deviceImg_out.get();
		isImageDeviceOwned = true;
	}

	GCProjectionDataDevice* dat_in = nullptr;
	bool isProjDataDeviceOwned = false;

	// In case the user provided a Host-side IProjectionData
	std::unique_ptr<GCProjectionDataDeviceOwned> deviceDat_in = nullptr;
	if (dat_in_const == nullptr)
	{
		auto* hostDat_in = dynamic_cast<const IProjectionData*>(in);
		ASSERT_MSG(
		    hostDat_in != nullptr,
		    "The Projection Data provded is not a GCProjectionDataDevice "
		    "nor a IProjectionData (host)");

		std::vector<const GCBinIterator*> binIterators;
		binIterators.push_back(binIter);  // We project only one subset
		deviceDat_in = std::make_unique<GCProjectionDataDeviceOwned>(
		    getScanner(), hostDat_in, binIterators);

		// Use owned GCProjectionDataDevice
		dat_in = deviceDat_in.get();
		isProjDataDeviceOwned = true;
	}
	else
	{
		dat_in = const_cast<GCProjectionDataDevice*>(dat_in_const);
		ASSERT_MSG(dat_in != nullptr,
		           "GCProjectionDataDevice is null. Cast failed");
	}


	if (!isProjDataDeviceOwned)
	{
		// To avoid altering the originally provided projection data, write in
		// the intermediary buffer instead of the ProjData buffer provided
		applyAttenuationOnLoadedBatchIfNeeded(dat_in, false);

		GCProjectionDataDevice* dataToProject =
		    attImageForBackprojection != nullptr ? &getIntermediaryProjData() :
		                                           dat_in;

		std::cout << "Backprojecting current batch..." << std::endl;
		applyOnLoadedBatch<false>(dataToProject, img_out);
		std::cout << "Done backprojecting current batch." << std::endl;
	}
	else
	{
		// Iterate over all the batches of the current subset
		const size_t numBatches = dat_in->getBatchSetup(0).getNumBatches();
		const ImageParams& imgParams = img_out->getParams();
		for (size_t batchId = 0; batchId < numBatches; batchId++)
		{
			std::cout << "Loading batch " << batchId + 1 << "..." << std::endl;
			dat_in->loadEventLORs(0, batchId, imgParams, getAuxStream());
			deviceDat_in->allocateForProjValues(getAuxStream());
			deviceDat_in->loadProjValuesFromReference(getAuxStream());
			std::cout << "Batch " << batchId + 1 << " loaded." << std::endl;
			applyAttenuationOnLoadedBatchIfNeeded(dat_in, dat_in, false);
			std::cout << "Backprojecting batch..." << std::endl;
			applyOnLoadedBatch<false>(dat_in, img_out);
			std::cout << "Done backprojecting batch." << std::endl;
		}
	}

	if (isImageDeviceOwned)
	{
		// Need to transfer the generated image back to the host
		deviceImg_out->transferToHostMemory(hostImg_out, false);
	}
}

void GCOperatorProjectorDD_gpu::applyAttenuationOnLoadedBatchIfNeeded(
    const GCProjectionDataDevice* imgProjData, bool duringForward)
{
	if (requiresIntermediaryProjData())
	{
		prepareIntermediaryBuffer(imgProjData);
		// Use Intermediary buffer as destination
		applyAttenuationOnLoadedBatchIfNeeded(
		    imgProjData, &getIntermediaryProjData(), duringForward);
	}
}

void GCOperatorProjectorDD_gpu::applyAttenuationOnLoadedBatchIfNeeded(
    const GCProjectionDataDevice* imgProjData,
    GCProjectionDataDevice* destProjData, bool duringForward)
{
	ImageDevice* attImageToUse;
	if (attImage != nullptr && duringForward)
	{
		attImageToUse = const_cast<ImageDevice*>(&getAttImageDevice());
	}
	else if (attImageForBackprojection != nullptr && !duringForward)
	{
		attImageToUse =
		    const_cast<ImageDevice*>(&getAttImageForBackprojectionDevice());
	}
	else
	{
		// Nothing to do
		return;
	}

	prepareIntermediaryBufferIfNeeded(imgProjData);
	std::cout << "Forward projecting current batch on attenuation image..."
	          << std::endl;

	applyOnLoadedBatch<true>(&getIntermediaryProjData(), attImageToUse);

	std::cout << "Done Forward projecting current batch on attenuation image."
	          << std::endl;

	applyAttenuationFactors(&getIntermediaryProjData(), imgProjData,
	                        destProjData, 0.1f);

	std::cout << "Done applying attenuation on current batch." << std::endl;
}

void GCOperatorProjectorDD_gpu::applyAdditiveOnLoadedBatchIfNeeded(
    GCProjectionDataDevice* imgProjData)
{
	if (addHisto != nullptr)
	{
		prepareIntermediaryBufferIfNeeded(imgProjData);
		std::cout << "Applying additive corrections on current batch..."
		          << std::endl;
		GCProjectionDataDevice& intermediaryBuffer = getIntermediaryProjData();
		intermediaryBuffer.loadProjValuesFromHost(imgProjData->getReference(),
		                                          addHisto, getAuxStream());
		imgProjData->addProjValues(&intermediaryBuffer, getMainStream());
		std::cout << "Done applying additive corrections on current batch..."
		          << std::endl;
	}
}

template <bool IsForward>
void GCOperatorProjectorDD_gpu::applyOnLoadedBatch(GCProjectionDataDevice* dat,
                                                   ImageDevice* img)
{
	setBatchSize(dat->getCurrentBatchSize());
	const auto cuScannerParams = getCUScannerParams(*getScanner());
	const auto cuImageParams = getCUImageParams(img->getParams());
	const GCTimeOfFlightHelper* tofHelperDevicePointer =
	    getTOFHelperDevicePointer();

	if (tofHelperDevicePointer == nullptr)
	{
		launchKernel<IsForward, false>(
		    dat->getProjValuesDevicePointer(), img->getDevicePointer(),
		    dat->getLorDet1PosDevicePointer(),
		    dat->getLorDet2PosDevicePointer(),
		    dat->getLorDet1OrientDevicePointer(),
		    dat->getLorDet2OrientDevicePointer(), nullptr, nullptr /*No TOF*/,
		    cuScannerParams, cuImageParams, getBatchSize(), getGridSize(),
		    getBlockSize(), getMainStream(), isSynchronized());
	}
	else
	{
		launchKernel<IsForward, true>(
		    dat->getProjValuesDevicePointer(), img->getDevicePointer(),
		    dat->getLorDet1PosDevicePointer(),
		    dat->getLorDet2PosDevicePointer(),
		    dat->getLorDet1OrientDevicePointer(),
		    dat->getLorDet2OrientDevicePointer(),
		    dat->getLorTOFValueDevicePointer(), tofHelperDevicePointer,
		    cuScannerParams, cuImageParams, getBatchSize(), getGridSize(),
		    getBlockSize(), getMainStream(), isSynchronized());
	}
}

void GCOperatorProjectorDD_gpu::applyAttenuationFactors(
    const GCProjectionDataDevice* attImgProj,
    const GCProjectionDataDevice* imgProj, GCProjectionDataDevice* destProj,
    float unitFactor)
{
	setBatchSize(destProj->getCurrentBatchSize());
	const cudaStream_t* stream = getAuxStream();
	const unsigned int gridSize = getGridSize();
	const unsigned int blockSize = getBlockSize();
	const bool synchronize = isSynchronized();
	const size_t batchSize = getBatchSize();
	if (stream != nullptr)
	{
		applyAttenuationFactors_kernel<<<gridSize, blockSize, 0, *stream>>>(
		    attImgProj->getProjValuesDevicePointer(),
		    imgProj->getProjValuesDevicePointer(),
		    destProj->getProjValuesDevicePointer(), unitFactor, batchSize);
		if (synchronize)
		{
			cudaStreamSynchronize(*stream);
		}
	}
	else
	{
		applyAttenuationFactors_kernel<<<gridSize, blockSize>>>(
		    attImgProj->getProjValuesDevicePointer(),
		    imgProj->getProjValuesDevicePointer(),
		    destProj->getProjValuesDevicePointer(), unitFactor, batchSize);
		if (synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}

template <bool IsForward, bool HasTOF>
void GCOperatorProjectorDD_gpu::launchKernel(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const GCTimeOfFlightHelper* pd_tofHelper, GCCUScannerParams scannerParams,
    GCCUImageParams imgParams, size_t batchSize, unsigned int gridSize,
    unsigned int blockSize, const cudaStream_t* stream, bool synchronize)
{
	ASSERT_MSG(pd_projValues != nullptr && pd_lorDet1Pos != nullptr &&
	               pd_lorDet2Pos != nullptr && pd_lorDet1Orient != nullptr &&
	               pd_lorDet2Orient != nullptr,
	           "Projection space not allocated on device");
	ASSERT_MSG(pd_image != nullptr, "Image space not allocated on device");

	if (stream != nullptr)
	{
		GCOperatorProjectorDDCU_kernel<IsForward, HasTOF>
		    <<<gridSize, blockSize, 0, *stream>>>(
		        pd_projValues, pd_image, pd_lorDet1Pos, pd_lorDet2Pos,
		        pd_lorDet1Orient, pd_lorDet2Orient, pd_lorTOFValue,
		        pd_tofHelper, scannerParams, imgParams, batchSize);
		if (synchronize)
		{
			cudaStreamSynchronize(*stream);
		}
	}
	else
	{
		GCOperatorProjectorDDCU_kernel<IsForward, HasTOF>
		    <<<gridSize, blockSize>>>(
		        pd_projValues, pd_image, pd_lorDet1Pos, pd_lorDet2Pos,
		        pd_lorDet1Orient, pd_lorDet2Orient, pd_lorTOFValue,
		        pd_tofHelper, scannerParams, imgParams, batchSize);
		if (synchronize)
		{
			cudaDeviceSynchronize();
		}
	}
	cudaCheckError();
}
