/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "omp.h"
#include <cstddef>

class GCGlobals
{
public:
	static int get_num_threads() { return num_threads; }
	static void set_num_threads(int t)
	{
		if (t <= 0)
			num_threads = omp_get_max_threads();
		else
			num_threads = t;
		omp_set_num_threads(t);
	}

private:
	static int num_threads;
};

class GCGlobalsCuda
{
public:
	static constexpr size_t maxVRAM_Allowed = 2ull<<30; // bytes
	static constexpr size_t threadsPerBlockData = 256;
	static constexpr size_t threadsPerBlockImg3d = 8;
	static constexpr size_t threadsPerBlockImg2d = 32;
};