/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/image/GCImageSpaceKernels.cuh"

__global__ void updateEM_kernel(const float* d_imgIn, float* d_imgOut,
                         const float* d_sensImg, const int nx, const int ny,
                         const int nz, const float EM_threshold)
{
	const long id_z = blockIdx.z * blockDim.z + threadIdx.z;
	const long id_y = blockIdx.y * blockDim.y + threadIdx.y;
	const long id_x = blockIdx.x * blockDim.x + threadIdx.x;
	const long pixelId = id_z * nx * ny + id_y * nx + id_x;

	if (id_z < nz && id_y < ny && id_x < nx)
	{
		if (d_sensImg[pixelId] > EM_threshold)
		{
			d_imgOut[pixelId] *= d_imgIn[pixelId] / d_sensImg[pixelId];
		}
	}
}

__global__ void applyThreshold_kernel(
    float* d_imgIn, const float* d_imgMask, const float threshold,
    const float val_le_scale, const float val_le_off, const float val_gt_scale,
    const float val_gt_off, const int nx, const int ny, const int nz)
{
	const long id_z = blockIdx.z * blockDim.z + threadIdx.z;
	const long id_y = blockIdx.y * blockDim.y + threadIdx.y;
	const long id_x = blockIdx.x * blockDim.x + threadIdx.x;
	const long pixelId = id_z * nx * ny + id_y * nx + id_x;

	if (id_z < nz && id_y < ny && id_x < nx)
	{
		if (d_imgMask[pixelId] <= threshold)
		{
			d_imgIn[pixelId] = d_imgIn[pixelId] * val_le_scale + val_le_off;
		}
		else
		{
			d_imgIn[pixelId] = d_imgIn[pixelId] * val_gt_scale + val_gt_off;
		}
	}
}

__global__ void setValue_kernel(float* d_imgIn, const float value, const int nx,
                                const int ny, const int nz)
{
	const long id_z = blockIdx.z * blockDim.z + threadIdx.z;
	const long id_y = blockIdx.y * blockDim.y + threadIdx.y;
	const long id_x = blockIdx.x * blockDim.x + threadIdx.x;
	const long pixelId = id_z * nx * ny + id_y * nx + id_x;

	if (id_z < nz && id_y < ny && id_x < nx)
	{
		d_imgIn[pixelId] = value;
	}
}

__global__ void addFirstImageToSecond_kernel(const float* d_imgIn,
                                             float* d_imgOut, int nx, int ny,
                                             int nz)
{
	const long id_z = blockIdx.z * blockDim.z + threadIdx.z;
	const long id_y = blockIdx.y * blockDim.y + threadIdx.y;
	const long id_x = blockIdx.x * blockDim.x + threadIdx.x;
	const long pixelId = id_z * nx * ny + id_y * nx + id_x;

	if (id_z < nz && id_y < ny && id_x < nx)
	{
		d_imgOut[pixelId] = d_imgOut[pixelId] + d_imgIn[pixelId];
	}
}
