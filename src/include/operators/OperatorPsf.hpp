/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/Image.hpp"
#include "operators/Operator.hpp"

#include <vector>

class OperatorPsf : public Operator
{
public:
	OperatorPsf(const ImageParams& img_params);
	OperatorPsf(const ImageParams& img_params,
	              const std::string& image_space_psf_filename);
	~OperatorPsf() override;

	void readFromFile(const std::string& image_space_psf_filename);

	void applyA(const Variable* in, Variable* out) override;
	void applyAH(const Variable* in, Variable* out) override;

	void convolve(const Image* in, Image* out,
	              const std::vector<float>& KernelX,
	              const std::vector<float>& KernelY,
	              const std::vector<float>& KernelZ) const;

protected:
	mutable std::vector<float> m_buffer_tmp;
	std::vector<int> m_kerSize;
	int m_nx;
	int m_ny;
	int m_nz;
	std::vector<float> m_KernelX;
	std::vector<float> m_KernelY;
	std::vector<float> m_KernelZ;
	std::vector<float> m_KernelX_flipped;
	std::vector<float> m_KernelY_flipped;
	std::vector<float> m_KernelZ_flipped;
	ImageParams m_params;
};
