/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/ImageBase.hpp"
#include "geometry/GCVector.hpp"
#include "utils/Array.hpp"

#include <string>

class Image : public ImageBase
{
public:
	Image(const ImageParams& img_params);
	~Image() override = default;

	Array3DBase<double>& getData();
	const Array3DBase<double>& getData() const;
	void copyFromImage(const Image* imSrc);
	void multWithScalar(double scalar);
	void addFirstImageToSecond(ImageBase* imCopy) const override;

	void setValue(double initValue) override;
	void applyThreshold(const ImageBase* maskImg, double threshold,
	                    double val_le_scale, double val_le_off,
	                    double val_gt_scale, double val_gt_off) override;
	void updateEMThreshold(ImageBase* updateImg, const ImageBase* normImg,
	                       double threshold) override;
	void writeToFile(const std::string& image_fname) const override;

	Array3DAlias<double> getArray() const;
	std::unique_ptr<Image> transformImage(const GCVector& rotation,
	                                        const GCVector& translation) const;

	double dot_product(Image* y) const;
	double interpol_image(GCVector pt);
	double interpol_image2(GCVector pt, Image* sens);
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

class ImageOwned : public Image
{
public:
	ImageOwned(const ImageParams& img_params);
	ImageOwned(const ImageParams& img_params, const std::string& filename);
	void allocate();
	void readFromFile(const std::string& image_file_name);
};

class ImageAlias : public Image
{
public:
	ImageAlias(const ImageParams& img_params);
	void bind(Array3DBase<double>& p_data);
};
