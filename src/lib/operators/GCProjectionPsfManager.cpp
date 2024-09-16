/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/GCProjectionPsfManager.hpp"

#include "utils/GCTools.hpp"

GCProjectionPsfManager::GCProjectionPsfManager(const std::string& psfFilename)
    : m_sStep(0.f), m_kSpacing(0.f)
{
	readFromFile(psfFilename);
}

void GCProjectionPsfManager::readFromFile(const std::string& psfFilename)
{
	Util::readCSV<float>(psfFilename, m_kernelDataRaw);
	m_sStep = m_kernelDataRaw[0][0];
	m_kSpacing = m_kernelDataRaw[0][1];
	int kernel_size = m_kernelDataRaw[0][2];
	int num_s = (m_kernelDataRaw.GetSize(1) - 3) / kernel_size;

	if ((kernel_size % 2) == 0)
	{
		throw std::runtime_error("Kernels must be of odd length");
	}

	m_kernels.Bind(m_kernelDataRaw[0] + 3, num_s, kernel_size);
	m_kernelsFlipped.allocate(m_kernels.GetSize(0), m_kernels.GetSize(1));
	for (size_t i = 0; i < m_kernels.GetSize(0); i++)
	{
		for (size_t j = 0; j < m_kernels.GetSize(1); j++)
		{
			m_kernelsFlipped[i][m_kernels.GetSize(1) - 1 - j] = m_kernels[i][j];
		}
	}
}


float GCProjectionPsfManager::getHalfWidth_mm() const
{
	// Zero-padding by one element
	return (m_kernels.GetSize(1) + 1) / 2 * m_kSpacing;
}

int GCProjectionPsfManager::getKernelSize() const
{
	return m_kernels.GetSize(1);
}

float* GCProjectionPsfManager::getKernel(const GCStraightLineParam& lor,
                                         bool flagFlipped) const
{
	const GCVector p1 = lor.point1;
	const GCVector p2 = lor.point2;
	float n_plane_x = p2.y - p1.y;
	float n_plane_y = p1.x - p2.x;
	float n_plane_norm =
	    std::sqrt(n_plane_x * n_plane_x + n_plane_y * n_plane_y);
	size_t s_idx;
	if (n_plane_norm == 0)
	{
		s_idx = 0;
	}
	else
	{
		n_plane_x /= n_plane_norm;
		n_plane_y /= n_plane_norm;
		float s = std::abs(p1.x * n_plane_x + p1.y * n_plane_y);
		s_idx = std::min(static_cast<size_t>(std::floor(s / m_sStep)),
		                 m_kernels.GetSize(0) - 1);
	}

	if (!flagFlipped)
	{
		return m_kernels[s_idx];
	}
	else
	{
		return m_kernelsFlipped[s_idx];
	}
}

float GCProjectionPsfManager::getWeight(const float* kernel, const float x0,
                                        const float x1) const
{
	int halfWidth = (m_kernels.GetSize(1) + 1) / 2;
	if (x0 > halfWidth * m_kSpacing || x1 < -halfWidth * m_kSpacing || x0 >= x1)
	{
		return 0.f;
	}
	return Util::calculateIntegral(kernel, m_kernels.GetSize(1), m_kSpacing, x0,
	                               x1);
}
