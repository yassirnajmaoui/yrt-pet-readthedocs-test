/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "GCAssert.hpp"

#include <vector_types.h>

class GCGPUBatchSetup
{
public:
	GCGPUBatchSetup(size_t totalSize, size_t maxBatchSize)
	{
		ASSERT(maxBatchSize > 0);
		// Note: totalSize and maxBatchSize must have the same units
		numBatches = totalSize / maxBatchSize + 1;
		lastBatchSize = totalSize % maxBatchSize;
		batchSize = maxBatchSize;

		// Special case for when there's only one batch
		if (numBatches == 1)
		{
			batchSize = lastBatchSize;
		}
		// Special case for when batches fit perfectly
		if (lastBatchSize == 0)
		{
			numBatches--;
			lastBatchSize = batchSize;
		}
	}
	[[nodiscard]] size_t getBatchSize(size_t batchId) const
	{
		if (batchId == numBatches - 1)
		{
			return lastBatchSize;
		}
		return batchSize;
	}
	[[nodiscard]] size_t getNumBatches() const { return numBatches; }

private:
	size_t batchSize;
	size_t lastBatchSize;
	size_t numBatches;
};

struct GCGPULaunchParams
{
	unsigned int gridSize;
	unsigned int blockSize;
};

struct GCGPULaunchParams3D
{
	dim3 gridSize;
	dim3 blockSize;
};
