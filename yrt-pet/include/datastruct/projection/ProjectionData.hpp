/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/projection/BinIterator.hpp"
#include "recon/Variable.hpp"
#include "utils/Types.hpp"

#include <functional>
#include <memory>

class ProjectionData : public Variable
{
public:
	~ProjectionData() override = default;

	// Mandatory methods
	virtual size_t count() const = 0;
	virtual float getProjectionValue(bin_t id) const = 0;
	virtual void setProjectionValue(bin_t id, float val) = 0;
	virtual det_id_t getDetector1(bin_t id) const = 0;
	virtual det_id_t getDetector2(bin_t id) const = 0;
	virtual det_pair_t getDetectorPair(bin_t id) const;
	virtual histo_bin_t getHistogramBin(bin_t bin) const;
	virtual std::unique_ptr<BinIterator> getBinIter(int numSubsets,
	                                                  int idxSubset) const = 0;

	// Optional methods
	virtual timestamp_t getTimestamp(bin_t id) const;
	virtual frame_t getFrame(bin_t id) const;
	virtual bool isUniform() const;
	virtual float getRandomsEstimate(bin_t id) const;
	// Time-of-flight
	virtual bool hasTOF() const;
	virtual float getTOFValue(bin_t id) const;
	// For motion correction
	virtual bool hasMotion() const;
	virtual size_t getNumFrames() const;
	virtual transform_t getTransformOfFrame(frame_t frame) const;
	// Special case when the LOR is not defined directly from the scanner's LUT
	virtual bool hasArbitraryLORs() const;
	virtual line_t getArbitraryLOR(bin_t id) const;

	// Helper functions
	virtual void clearProjections(float value);
	virtual void divideMeasurements(const ProjectionData* measurements,
	                                const BinIterator* binIter);

	virtual void operationOnEachBin(const std::function<float(bin_t)>& func);
	// Note: The function given as argument should be able to be called in
	// parallel without race conditions for different bins.
	// In other words, two different bins shouldn't point
	// to the same memory location.
	virtual void
	    operationOnEachBinParallel(const std::function<float(bin_t)>& func);
};
