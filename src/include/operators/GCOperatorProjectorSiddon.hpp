/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/GCMultiRayGenerator.hpp"
#include "operators/GCOperatorProjector.hpp"

#include "omp.h"

class Image;

class GCOperatorProjectorSiddon : public GCOperatorProjector
{
public:
	GCOperatorProjectorSiddon(const GCOperatorProjectorParams& p_projParams);

	double forwardProjection(const Image* img, const IProjectionData* dat,
	                         bin_t bin) override;

	void backProjection(Image* img, const IProjectionData* dat, bin_t bin,
	                    double projValue) override;


	// Projection
	double forwardProjection(const Image* img, const GCStraightLineParam& lor,
	                         const GCVector& n1, const GCVector& n2,
	                         const GCTimeOfFlightHelper* tofHelper = nullptr,
	                         float tofValue = 0.f) const;
	void backProjection(Image* img, const GCStraightLineParam& lor,
	                    const GCVector& n1, const GCVector& n2,
	                    double projValue,
	                    const GCTimeOfFlightHelper* tofHelper = nullptr,
	                    float tofValue = 0.f) const;

	// Without Multi-ray siddon
	static double singleForwardProjection(
	    const Image* img, const GCStraightLineParam& lor,
	    const GCTimeOfFlightHelper* tofHelper = nullptr, float tofValue = 0.f);
	static void singleBackProjection(
	    Image* img, const GCStraightLineParam& lor, double projValue,
	    const GCTimeOfFlightHelper* tofHelper = nullptr, float tofValue = 0.f);


	template <bool IS_FWD, bool FLAG_INCR, bool FLAG_TOF>
	static void project_helper(Image* img, const GCStraightLineParam& lor,
	                           double& value,
	                           const GCTimeOfFlightHelper* tofHelper = nullptr,
	                           float tofValue = 0.f);

	int getNumRays() const;
	void setNumRays(int n);

private:
	int m_numRays;
	std::unique_ptr<std::vector<GCMultiRayGenerator>> mp_lineGen;
};
