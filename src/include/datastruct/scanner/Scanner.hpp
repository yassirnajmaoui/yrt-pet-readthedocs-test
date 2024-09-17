/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/scanner/DetCoord.hpp"
#include "geometry/GCVector.hpp"

#include <filesystem>
#include <string>

#define GCSCANNER_FILE_VERSION 3.0

namespace fs = std::filesystem;

class Scanner
{
public:
	size_t getNumDets() const;
	size_t getTheoreticalNumDets() const;
	GCVector getDetectorPos(det_id_t id) const;
	GCVector getDetectorOrient(det_id_t id) const;
	const DetectorSetup* getDetectorSetup() const { return mp_detectors; }
	// Allocate and fill array with detector positions
	void createLUT(Array2D<float>& lut) const;

protected:
	Scanner();

public:
	std::string scannerName;
	float axialFOV, crystalSize_z, crystalSize_trans, crystalDepth,
	    scannerRadius;
	float collimatorRadius, fwhm, energyLLD;  // for Scatter

	// dets_per_ring : Number of detectors per ring (not counting DOI)
	// num_rings : Number of rings in total (not countring DOI)
	// num_doi : Number of DOI crystals (ex: 2 for SAVANT)
	// max_ring_diff : Maximum ring difference (number of rings)
	// min_ang_diff : Minimum angular difference, in terms of detector indices
	size_t dets_per_ring, num_rings, num_doi, max_ring_diff, min_ang_diff;
	size_t dets_per_block;

protected:
	// Base class that encapsulates the calculations for both a regular and a
	// LUT based scanner
	DetectorSetup* mp_detectors;
};

// Owned class for when the detector setup is of the right ownership
class ScannerOwned : public Scanner
{
public:
	ScannerOwned();
	ScannerOwned(const std::string& p_definitionFile);
	void readFromFile(const std::string& p_definitionFile);
	void readFromString(const std::string& fileContents);
	std::string getScannerPath() const;
	void setScannerPath(const fs::path& p);
	void setScannerPath(const std::string& s);

protected:
	fs::path m_scannerPath;
	std::unique_ptr<DetectorSetup> mp_detectorsPtr;
};

// Alias scanner class for when the detector setup is outside the scope
class ScannerAlias : public Scanner
{
public:
	ScannerAlias();
	void setDetectorSetup(DetectorSetup* d) { this->mp_detectors = d; }
};
