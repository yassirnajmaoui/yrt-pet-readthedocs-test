/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "recon/GCVariable.hpp"

#include "nlohmann/json_fwd.hpp"
#include <string>

#define GCIMAGEPARAMS_FILE_VERSION 1.0

class GCImageParams
{
public:
	int nx;
	int ny;
	int nz;
	double length_x;
	double length_y;
	double length_z;
	double off_x;
	double off_y;
	double off_z;

	// Automatically populated fields
	double vx, vy, vz;
	float fov_radius;

	GCImageParams();
	GCImageParams(int nxi, int nyi, int nzi, double length_xi, double length_yi,
	              double length_zi, double offset_xi = 0.,
	              double offset_yi = 0., double offset_zi = 0.);
	GCImageParams(const GCImageParams& in);
	GCImageParams& operator=(const GCImageParams& in);
	explicit GCImageParams(const std::string& fname);
	bool isSameDimensionsAs(const GCImageParams& other) const;
	bool isSameLengthsAs(const GCImageParams& other) const;
	bool isSameOffsetsAs(const GCImageParams& other) const;
	bool isSameAs(const GCImageParams& other) const;

	void copy(const GCImageParams& in);
	void setup();
	void serialize(const std::string& fname) const;
	void writeToJSON(nlohmann::json& geom_json) const;
	void deserialize(const std::string& fname);
	void readFromJSON(nlohmann::json& geom_json);
	bool isValid() const;
};

class GCImageBase : public GCVariable
{
public:
	GCImageBase(const GCImageParams& imgParams);
	~GCImageBase() override = default;

	// Common functions
	float getRadius() const;
	const GCImageParams& getParams() const;
	void setParams(const GCImageParams& newParams);

	virtual void setValue(double initValue) = 0;
	virtual void addFirstImageToSecond(GCImageBase* second) const = 0;
	virtual void applyThreshold(const GCImageBase* mask_img, double threshold,
	                            double val_le_scale, double val_le_off,
	                            double val_gt_scale, double val_gt_off) = 0;
	virtual void writeToFile(const std::string& image_fname) const = 0;
	virtual void updateEMThreshold(GCImageBase* update_img,
	                               const GCImageBase* norm_img,
	                               double threshold) = 0;

private:
	GCImageParams m_params;
};
