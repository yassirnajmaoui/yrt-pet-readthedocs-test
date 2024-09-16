/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

typedef struct GCCUScannerParams
{
	float crystalSize_trans;
	float crystalSize_z;
	size_t numDets;
} GCCUScannerParams;

typedef struct GCCUImageParams
{
	int voxelNumber[3];
	float voxelSize[3];
	float imgLength[3];
	float offset[3];
} GCCUImageParams;
