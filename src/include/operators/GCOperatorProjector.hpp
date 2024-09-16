/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "operators/GCOperator.hpp"
#include "operators/GCProjectionPsfManager.hpp"
#include "operators/GCTimeOfFlight.hpp"
#include "utils/GCTypes.hpp"

class BinIterator;
class Image;
class GCScanner;
class IProjectionData;
class IHistogram;

class GCOperatorProjectorParams
{
public:
	GCOperatorProjectorParams(const BinIterator* p_binIter,
	                          const GCScanner* p_scanner,
	                          float p_tofWidth_ps = 0.f, int p_tofNumStd = 0,
	                          std::string p_psfProjFilename = "",
	                          int p_num_rays = 1);

	const BinIterator* binIter;
	const GCScanner* scanner;

	// Time of Flight
	float tofWidth_ps;
	int tofNumStd;

	// Projection-domain PSF
	std::string psfProjFilename;

	// Multi-ray siddon only
	int numRays;
};

// Device-agnostic virtual class
class GCOperatorProjectorBase : public GCOperator
{
public:
	struct ProjectionProperties
	{
		GCStraightLineParam lor;
		float tofValue;
		float randomsEstimate;
		GCVector det1Orient;
		GCVector det2Orient;
	};

	GCOperatorProjectorBase(const GCOperatorProjectorParams& p_projParams);

	const GCScanner* getScanner() const;
	const BinIterator* getBinIter() const;

	virtual void setAttImage(const Image* p_attImage);  // alias
	virtual void setAttImageForBackprojection(const Image* p_attImage);
	void setAttenuationImage(const Image* p_attImage);
	virtual void setAddHisto(const IHistogram* p_addHisto);
	void setBinIter(const BinIterator* p_binIter);

	const Image* getAttImage() const;
	const Image* getAttImageForBackprojection() const;
	const IHistogram* getAddHisto() const;

protected:
	// Bin iterator
	const BinIterator* binIter;

	// To take scanner properties into account
	const GCScanner* scanner;

	// Attenuation image for forward projection (when there is motion)
	const Image* attImage;
	// For generating a sensitivity image with attenuation correction
	const Image* attImageForBackprojection;
	// Additive histogram
	const IHistogram* addHisto;
};

class GCOperatorProjector : public GCOperatorProjectorBase
{
public:
	enum ProjectorType
	{
		SIDDON = 0,
		DD,
		DD_GPU
	};

	GCOperatorProjector(const GCOperatorProjectorParams& p_projParams);

	// Virtual functions
	virtual double forwardProjection(const Image* in_image,
	                                 const IProjectionData* dat, bin_t bin) = 0;
	virtual void backProjection(Image* in_image, const IProjectionData* dat,
	                            bin_t bin, double projValue) = 0;

	void applyA(const GCVariable* in, GCVariable* out) override;
	void applyAH(const GCVariable* in, GCVariable* out) override;

	void setupTOFHelper(float tofWidth_ps, int tofNumStd = -1);
	void setupProjPsfManager(const std::string& psfFilename);

	const GCTimeOfFlightHelper* getTOFHelper() const;
	const GCProjectionPsfManager* getProjectionPsfManager() const;

	static void get_alpha(double r0, double r1, double p1, double p2,
	                      double inv_p12, double& amin, double& amax);

protected:
	// Time of flight
	std::unique_ptr<GCTimeOfFlightHelper> mp_tofHelper;

	// Projection-domain PSF
	std::unique_ptr<GCProjectionPsfManager> mp_projPsfManager;
};
