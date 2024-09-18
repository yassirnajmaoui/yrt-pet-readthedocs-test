/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/StraightLineParam.hpp"
#include "utils/Array.hpp"

class ProjectionPsfManager
{
public:
	explicit ProjectionPsfManager(const std::string& psfFilename);
	void readFromFile(const std::string& psfFilename);
	float getHalfWidth_mm() const;
	float getWeight(const float* kernel, float x0, float x1) const;
	float* getKernel(const StraightLineParam& lor,
	                 bool flagFlipped = false) const;
	int getKernelSize() const;

protected:
	Array2D<float> m_kernelDataRaw;
	Array2DAlias<float> m_kernels;
	Array2D<float> m_kernelsFlipped;
	float m_sStep;
	float m_kSpacing;
};
