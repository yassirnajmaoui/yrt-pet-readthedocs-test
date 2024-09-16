/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/GCImageBase.hpp"
#include "geometry/GCVector.hpp"
#include "utils/Array.hpp"

#include <string>

class GCImage : public GCImageBase
{
public:
	GCImage(const GCImageParams& img_params);
	~GCImage() override = default;

	Array3DBase<double>& getData();
	const Array3DBase<double>& getData() const;
	void copyFromImage(const GCImage* imSrc);
	void multWithScalar(double scalar);
	void addFirstImageToSecond(GCImageBase* imCopy) const override;

	void setValue(double initValue) override;
	void applyThreshold(const GCImageBase* maskImg, double threshold,
	                    double val_le_scale, double val_le_off,
	                    double val_gt_scale, double val_gt_off) override;
	void updateEMThreshold(GCImageBase* updateImg, const GCImageBase* normImg,
	                       double threshold) override;
	void writeToFile(const std::string& image_fname) const override;

	Array3DAlias<double> getArray() const;
	std::unique_ptr<GCImage> transformImage(const GCVector& rotation,
	                                        const GCVector& translation) const;

	double dot_product(GCImage* y) const;
	double interpol_image(GCVector pt);
	double interpol_image2(GCVector pt, GCImage* sens);
	double nearest_neigh(const GCVector& pt) const;
	double nearest_neigh2(const GCVector& pt, int* pi, int* pj, int* pk) const;
	void update_image_nearest_neigh(const GCVector& pt, double value,
	                                bool mult_flag);
	void assign_image_nearest_neigh(const GCVector& pt, double value);
	bool get_nearest_neigh_idx(const GCVector& pt, int* pi, int* pj,
	                           int* pk) const;
	void update_image_inter(GCVector point, double value, bool mult_flag);
	void assign_image_inter(GCVector point, double value);
	bool get_voxel_ind(const GCVector& point, int* i, int* j, int* k) const;
	bool get_voxel_ind(const GCVector& point, double* i, double* j,
	                   double* k) const;

protected:
	std::unique_ptr<Array3DBase<double>> m_dataPtr;
};

class GCImageOwned : public GCImage
{
public:
	GCImageOwned(const GCImageParams& img_params);
	GCImageOwned(const GCImageParams& img_params, const std::string& filename);
	void allocate();
	void readFromFile(const std::string& image_file_name);
};

class GCImageAlias : public GCImage
{
public:
	GCImageAlias(const GCImageParams& img_params);
	void bind(Array3DBase<double>& p_data);
};
