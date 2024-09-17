/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "utils/GCDeviceArray.cuh"

#include <memory>

class Scanner;

class ScannerDevice
{
public:
	// Loads the scanner into the device
	explicit ScannerDevice(const Scanner* pp_scanner,
	                         const cudaStream_t* pp_stream = nullptr);
	void allocate(const cudaStream_t* stream = nullptr);
	void load(const cudaStream_t* stream = nullptr);
	const float4* getDetPosDevicePointer() const;
	const float4* getDetOrientDevicePointer() const;
	const Scanner* getScanner() const;

private:
	std::unique_ptr<GCDeviceArray<float4>> mpd_detPos;
	std::unique_ptr<GCDeviceArray<float4>> mpd_detOrient;
	const Scanner* mp_scanner;
	bool isAllocated;
	bool isLoaded;
};