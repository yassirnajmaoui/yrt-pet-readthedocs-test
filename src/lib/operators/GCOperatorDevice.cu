/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/GCOperatorDevice.cuh"

#include "datastruct/image/Image.hpp"
#include "datastruct/scanner/GCScanner.hpp"
#include "utils/GCGPUUtils.cuh"
#include "utils/GCGlobals.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;

void py_setup_gcoperatorprojectordevice(py::module& m)
{
	auto c = py::class_<GCOperatorProjectorDevice, GCOperatorProjectorBase>(
	    m, "GCOperatorProjectorDevice");
}
#endif

namespace Util
{
	GCGPULaunchParams3D initiateDeviceParameters(const ImageParams& params)
	{
		GCGPULaunchParams3D launchParams;
		if (params.nz > 1)
		{
			const size_t threadsPerBlockDimImage =
			    GCGlobalsCuda::threadsPerBlockImg3d;
			const auto threadsPerBlockDimImage_float =
			    static_cast<float>(threadsPerBlockDimImage);
			const auto threadsPerBlockDimImage_uint =
			    static_cast<unsigned int>(threadsPerBlockDimImage);

			launchParams.gridSize = {
			    static_cast<unsigned int>(
			        std::ceil(params.nx / threadsPerBlockDimImage_float)),
			    static_cast<unsigned int>(
			        std::ceil(params.ny / threadsPerBlockDimImage_float)),
			    static_cast<unsigned int>(
			        std::ceil(params.nz / threadsPerBlockDimImage_float))};

			launchParams.blockSize = {threadsPerBlockDimImage_uint,
			                          threadsPerBlockDimImage_uint,
			                          threadsPerBlockDimImage_uint};
		}
		else
		{
			const size_t threadsPerBlockDimImage =
			    GCGlobalsCuda::threadsPerBlockImg2d;
			const auto threadsPerBlockDimImage_float =
			    static_cast<float>(threadsPerBlockDimImage);
			const auto threadsPerBlockDimImage_uint =
			    static_cast<unsigned int>(threadsPerBlockDimImage);

			launchParams.gridSize = {
			    static_cast<unsigned int>(
			        std::ceil(params.nx / threadsPerBlockDimImage_float)),
			    static_cast<unsigned int>(
			        std::ceil(params.ny / threadsPerBlockDimImage_float)),
			    1};

			launchParams.blockSize = {threadsPerBlockDimImage_uint,
			                          threadsPerBlockDimImage_uint, 1};
		}
		return launchParams;
	}

	GCGPULaunchParams initiateDeviceParameters(size_t batchSize)
	{
		GCGPULaunchParams launchParams{};
		launchParams.gridSize = static_cast<unsigned int>(
		    std::ceil(batchSize /
		              static_cast<float>(GCGlobalsCuda::threadsPerBlockData)));
		launchParams.blockSize = GCGlobalsCuda::threadsPerBlockData;
		return launchParams;
	}
}  // namespace Util

GCCUScannerParams GCOperatorDevice::getCUScannerParams(const GCScanner& scanner)
{
	GCCUScannerParams params;
	params.crystalSize_trans = scanner.crystalSize_trans;
	params.crystalSize_z = scanner.crystalSize_z;
	params.numDets = scanner.getNumDets();
	return params;
}

GCCUImageParams
    GCOperatorDevice::getCUImageParams(const ImageParams& imgParams)
{
	GCCUImageParams params;

	params.voxelNumber[0] = imgParams.nx;
	params.voxelNumber[1] = imgParams.ny;
	params.voxelNumber[2] = imgParams.nz;

	params.imgLength[0] = static_cast<float>(imgParams.length_x);
	params.imgLength[1] = static_cast<float>(imgParams.length_y);
	params.imgLength[2] = static_cast<float>(imgParams.length_z);

	params.voxelSize[0] = static_cast<float>(imgParams.vx);
	params.voxelSize[1] = static_cast<float>(imgParams.vy);
	params.voxelSize[2] = static_cast<float>(imgParams.vz);

	params.offset[0] = static_cast<float>(imgParams.off_x);
	params.offset[1] = static_cast<float>(imgParams.off_y);
	params.offset[2] = static_cast<float>(imgParams.off_z);

	return params;
}

GCOperatorProjectorDevice::GCOperatorProjectorDevice(
    const GCOperatorProjectorParams& projParams, bool p_synchronized,
    const cudaStream_t* pp_mainStream, const cudaStream_t* pp_auxStream)
    : GCOperatorProjectorBase(projParams), GCOperatorDevice()
{
	if (projParams.tofWidth_ps > 0.f)
	{
		setupTOFHelper(projParams.tofWidth_ps, projParams.tofNumStd);
	}

	m_batchSize = 0ull;
	m_synchonized = p_synchronized;
	mp_mainStream = pp_mainStream;
	mp_auxStream = pp_auxStream;
}

unsigned int GCOperatorProjectorDevice::getGridSize() const
{
	return m_launchParams.gridSize;
}
unsigned int GCOperatorProjectorDevice::getBlockSize() const
{
	return m_launchParams.blockSize;
}

bool GCOperatorProjectorDevice::isSynchronized() const
{
	return m_synchonized;
}

const cudaStream_t* GCOperatorProjectorDevice::getMainStream() const
{
	return mp_mainStream;
}

const cudaStream_t* GCOperatorProjectorDevice::getAuxStream() const
{
	return mp_auxStream;
}

void GCOperatorProjectorDevice::setBatchSize(size_t newBatchSize)
{
	m_batchSize = newBatchSize;
	m_launchParams = Util::initiateDeviceParameters(m_batchSize);
}

ProjectionDataDeviceOwned&
    GCOperatorProjectorDevice::getIntermediaryProjData()
{
	ASSERT_MSG(mp_intermediaryProjData != nullptr,
	           "Projection-space GPU Intermediary buffer not initialized");
	return *mp_intermediaryProjData;
}

const ImageDevice& GCOperatorProjectorDevice::getAttImageDevice() const
{
	ASSERT_MSG(mp_attImageDevice != nullptr,
	           "Device attenuation image not initialized");
	return *mp_attImageDevice;
}

const ImageDevice&
    GCOperatorProjectorDevice::getAttImageForBackprojectionDevice() const
{
	ASSERT_MSG(mp_attImageForBackprojectionDevice != nullptr,
	           "Device attenuation image for backprojection not initialized");
	return *mp_attImageForBackprojectionDevice;
}

size_t GCOperatorProjectorDevice::getBatchSize() const
{
	return m_batchSize;
}

void GCOperatorProjectorDevice::setAttImage(const Image* attImage)
{
	GCOperatorProjectorBase::setAttImage(attImage);

	mp_attImageDevice = std::make_unique<ImageDeviceOwned>(
	    attImage->getParams(), getAuxStream());
	mp_attImageDevice->allocate(getAuxStream());
	mp_attImageDevice->transferToDeviceMemory(attImage, false);
}

void GCOperatorProjectorDevice::setAttImageForBackprojection(
    const Image* attImage)
{
	GCOperatorProjectorBase::setAttImageForBackprojection(attImage);

	mp_attImageForBackprojectionDevice = std::make_unique<ImageDeviceOwned>(
	    attImage->getParams(), getAuxStream());
	mp_attImageForBackprojectionDevice->allocate(getAuxStream());
	mp_attImageForBackprojectionDevice->transferToDeviceMemory(attImage, false);
}

void GCOperatorProjectorDevice::setAddHisto(const Histogram* p_addHisto)
{
	GCOperatorProjectorBase::setAddHisto(p_addHisto);
}

void GCOperatorProjectorDevice::setupTOFHelper(float tofWidth_ps, int tofNumStd)
{
	mp_tofHelper = std::make_unique<GCDeviceObject<GCTimeOfFlightHelper>>(
	    tofWidth_ps, tofNumStd);
}

bool GCOperatorProjectorDevice::requiresIntermediaryProjData() const
{
	// We need an intermediary projectorParam if we'll need to do attenuation
	// correction or additive correction (scatter/randoms)
	return attImage != nullptr || attImageForBackprojection != nullptr ||
	       addHisto != nullptr;
}

void GCOperatorProjectorDevice::prepareIntermediaryBufferIfNeeded(
    const ProjectionDataDevice* orig)
{
	if (requiresIntermediaryProjData())
	{
		prepareIntermediaryBuffer(orig);
	}
}

void GCOperatorProjectorDevice::prepareIntermediaryBuffer(
    const ProjectionDataDevice* orig)
{
	if (mp_intermediaryProjData == nullptr)
	{
		mp_intermediaryProjData =
		    std::make_unique<ProjectionDataDeviceOwned>(orig);
	}
	mp_intermediaryProjData->allocateForProjValues(getAuxStream());
}

const GCTimeOfFlightHelper*
	GCOperatorProjectorDevice::getTOFHelperDevicePointer() const
{
	if(mp_tofHelper != nullptr)
	{
		return mp_tofHelper->getDevicePointer();
	}
	return nullptr;
}
