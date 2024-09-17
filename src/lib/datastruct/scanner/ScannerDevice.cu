/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/scanner/ScannerDevice.cuh"

#include "datastruct/scanner/Scanner.hpp"
#include "utils/GCPageLockedBuffer.cuh"

ScannerDevice::ScannerDevice(const Scanner* pp_scanner,
                                 const cudaStream_t* pp_stream)
    : mp_scanner(pp_scanner), isAllocated(false), isLoaded(false)
{
	mpd_detPos = std::make_unique<GCDeviceArray<float4>>();
	mpd_detOrient = std::make_unique<GCDeviceArray<float4>>();
	load(pp_stream);
}

void ScannerDevice::load(const cudaStream_t* stream)
{
	if (!isAllocated)
	{
		allocate(stream);
	}

	// We use a single large buffer to store two sets of data
	const size_t numDets = mp_scanner->getNumDets();
	GCPageLockedBuffer<float4> tempBuffer(numDets * 2);
	float4* ph_detPos = tempBuffer.getPointer();
	float4* ph_detOrient = ph_detPos + numDets;

	const DetectorSetup* detectorSetup = mp_scanner->getDetectorSetup();

#pragma omp parallel for default(none) \
    firstprivate(ph_detPos, ph_detOrient, numDets, detectorSetup)
	for (size_t id_det = 0; id_det < numDets; id_det++)
	{
		ph_detPos[id_det].x = detectorSetup->getXpos(id_det);
		ph_detPos[id_det].y = detectorSetup->getYpos(id_det);
		ph_detPos[id_det].z = detectorSetup->getZpos(id_det);
		ph_detOrient[id_det].x = detectorSetup->getXorient(id_det);
		ph_detOrient[id_det].y = detectorSetup->getYorient(id_det);
		ph_detOrient[id_det].z = detectorSetup->getZorient(id_det);
	}

	mpd_detOrient->copyFromHost(ph_detOrient, numDets, stream);
	mpd_detPos->copyFromHost(ph_detPos, numDets, stream);
	isLoaded = true;
}

void ScannerDevice::allocate(const cudaStream_t* stream)
{
	const size_t numDets = mp_scanner->getNumDets();
	mpd_detPos->allocate(numDets, stream);
	mpd_detOrient->allocate(numDets, stream);
	isAllocated = true;
}

const float4* ScannerDevice::getDetPosDevicePointer() const
{
	return mpd_detPos->getDevicePointer();
}
const float4* ScannerDevice::getDetOrientDevicePointer() const
{
	return mpd_detOrient->getDevicePointer();
}

const Scanner* ScannerDevice::getScanner() const
{
	return mp_scanner;
}
