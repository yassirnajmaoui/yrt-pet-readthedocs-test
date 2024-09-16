/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "operators/GCTimeOfFlight.hpp"
#include "recon/GCCUParameters.hpp"

#ifdef __CUDACC__

__device__ float3 operator+(const float3& a, const float3& b);
__device__ float3 operator-(const float3& a, const float3& b);
__device__ float3 operator*(const float3& a, float f);
__device__ float3 operator/(const float3& a, float f);

__device__ float4 operator+(const float4& a, const float4& b);
__device__ float4 operator-(const float4& a, const float4& b);
__device__ float4 operator*(const float4& a, float f);

__device__ float norm(const float3& a);
__device__ float3 cross(float3 a, float3 b);

__global__ void gatherLORs_kernel(const uint2* pd_lorDetsId,
                                  const float4* pd_detsPos,
                                  const float4* pd_detsOrient,
                                  float4* pd_lorDet1Pos, float4* pd_lorDet2Pos,
                                  float4* pd_lorDet1Orient,
                                  float4* pd_lorDet2Orient,
                                  GCCUImageParams imgParams, size_t batchSize);

template <bool IsForward, bool HasTOF>
__global__ void GCOperatorProjectorDDCU_kernel(
    float* pd_projValues, float* pd_image, const float4* pd_lorDet1Pos,
    const float4* pd_lorDet2Pos, const float4* pd_lorDet1Orient,
    const float4* pd_lorDet2Orient, const float* pd_lorTOFValue,
    const GCTimeOfFlightHelper* pd_tofHelper, GCCUScannerParams scannerParams,
    GCCUImageParams imgParams, size_t batchSize);


__global__ void applyAttenuationFactors_kernel(const float* pd_attImgProjData,
                                               const float* pd_imgProjData,
                                               float* pd_destProjData,
                                               float unitFactor,
                                               size_t maxNumberOfEvents);

#endif
