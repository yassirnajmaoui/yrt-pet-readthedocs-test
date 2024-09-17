/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "catch.hpp"

#include "utils/GPUUtils.cuh"
#include <iostream>
#include <memory>

__global__ void cudaSum(float* a, float* b, float* c, int n)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
	{
		c[tid] = a[tid] + b[tid];
	}
}

TEST_CASE("cuda", "[cuda]")
{
	SECTION("test-simple")
	{
		int n = 100;
		float* a = new float[n];
		float* b = new float[n];
		float* c = new float[n];
		float* c_ref = new float[n];

		for (int i = 0; i < n; i++)
		{
			a[i] = rand() / (float)RAND_MAX;
			b[i] = rand() / (float)RAND_MAX;
			c_ref[i] = a[i] + b[i];
		}
		float* a_g = nullptr;
		gpuErrchk(cudaMalloc(&a_g, n * sizeof(float)));
		float* b_g = nullptr;
		gpuErrchk(cudaMalloc(&b_g, n * sizeof(float)));
		float* c_g = nullptr;
		gpuErrchk(cudaMalloc(&c_g, n * sizeof(float)));

		gpuErrchk(cudaMemcpy((char*)a_g, (char*)a, n * sizeof(float),
		                     cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy((char*)b_g, (char*)b, n * sizeof(float),
		                     cudaMemcpyHostToDevice));

		int block_size = 32;
		int num_blocks = n / block_size + 1;
		cudaSum<<<num_blocks, block_size>>>(a_g, b_g, c_g, n);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		gpuErrchk(cudaMemcpy((char*)c, (char*)c_g, n * sizeof(float),
		                     cudaMemcpyDeviceToHost));

		float max_err = 0.f;
		for (int i = 0; i < n; i++)
		{
			max_err = std::max(max_err, fabs(c_ref[i] - c[i]));
		}

		REQUIRE(max_err < 1e-6);

		gpuErrchk(cudaFree(a_g));
		gpuErrchk(cudaFree(b_g));
		gpuErrchk(cudaFree(c_g));
		delete[] a;
		delete[] b;
		delete[] c;
		delete[] c_ref;
	}
}
