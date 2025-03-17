/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/DeviceSynchronized.cuh"

#include "datastruct/image/Image.hpp"
#include "utils/GPUUtils.cuh"
#include "utils/Globals.hpp"


namespace Util
{
	GPULaunchParams3D initiateDeviceParameters(const ImageParams& params)
	{
		GPULaunchParams3D launchParams;
		if (params.nz > 1)
		{
			const size_t threadsPerBlockDimImage =
			    GlobalsCuda::ThreadsPerBlockImg3d;
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
			    GlobalsCuda::ThreadsPerBlockImg2d;
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

	GPULaunchParams initiateDeviceParameters(size_t batchSize)
	{
		GPULaunchParams launchParams{};
		launchParams.gridSize = static_cast<unsigned int>(std::ceil(
		    batchSize / static_cast<float>(GlobalsCuda::ThreadsPerBlockData)));
		launchParams.blockSize = GlobalsCuda::ThreadsPerBlockData;
		return launchParams;
	}
}  // namespace Util

const cudaStream_t* DeviceSynchronized::getMainStream() const
{
	return mp_mainStream;
}

const cudaStream_t* DeviceSynchronized::getAuxStream() const
{
	return mp_auxStream;
}

CUScannerParams DeviceSynchronized::getCUScannerParams(const Scanner& scanner)
{
	CUScannerParams params;
	params.crystalSize_trans = scanner.crystalSize_trans;
	params.crystalSize_z = scanner.crystalSize_z;
	params.numDets = scanner.getNumDets();
	return params;
}

CUImageParams DeviceSynchronized::getCUImageParams(const ImageParams& imgParams)
{
	CUImageParams params;

	params.voxelNumber[0] = imgParams.nx;
	params.voxelNumber[1] = imgParams.ny;
	params.voxelNumber[2] = imgParams.nz;

	params.imgLength[0] = imgParams.length_x;
	params.imgLength[1] = imgParams.length_y;
	params.imgLength[2] = imgParams.length_z;

	params.voxelSize[0] = imgParams.vx;
	params.voxelSize[1] = imgParams.vy;
	params.voxelSize[2] = imgParams.vz;

	params.offset[0] = imgParams.off_x;
	params.offset[1] = imgParams.off_y;
	params.offset[2] = imgParams.off_z;

	params.fovRadius = imgParams.fovRadius;

	return params;
}

DeviceSynchronized::DeviceSynchronized(const cudaStream_t* pp_mainStream,
                                       const cudaStream_t* pp_auxStream)
{
	if (pp_mainStream != nullptr)
	{
		mp_mainStream = pp_mainStream;
	}
	else
	{
		mp_mainStreamPtr = std::make_unique<GPUStream>();
		mp_mainStream = &mp_mainStreamPtr->getStream();
	}
	if (pp_auxStream != nullptr)
	{
		mp_auxStream = pp_auxStream;
	}
	else
	{
		mp_auxStreamPtr = std::make_unique<GPUStream>();
		mp_auxStream = &mp_auxStreamPtr->getStream();
	}
}
