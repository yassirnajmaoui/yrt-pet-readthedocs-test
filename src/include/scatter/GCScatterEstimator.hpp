/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/projection/GCHistogram3D.hpp"
#include "scatter/GCCrystal.hpp"
#include "scatter/GCSingleScatterSimulator.hpp"

class GCScanner;
class GCImage;

namespace Scatter
{
	class GCScatterEstimator
	{
	public:
		GCScatterEstimator(const GCScanner& pr_scanner,
		                   const GCImage& pr_lambda, const GCImage& pr_mu,
		                   const GCHistogram3D* pp_promptsHis,
		                   const GCHistogram3D* pp_normOrSensHis,
		                   const GCHistogram3D* pp_randomsHis,
		                   const GCHistogram3D* pp_acfHis,
		                   CrystalMaterial p_crystalMaterial, int seedi,
		                   bool p_doTailFitting, bool isNorm, int maskWidth,
		                   float maskThreshold, bool saveIntermediary);

		void estimateScatter(size_t numberZ, size_t numberPhi, size_t numberR,
		                     bool printProgress = false);

		const GCHistogram3DOwned* getScatterHistogram() const;

	protected:
		static void generateScatterTailsMask(const GCHistogram3D& acfHis,
		                                     std::vector<bool>& mask,
		                                     size_t maskWidth,
		                                     float maskThreshold);

		void saveScatterTailsMask();

	private:
		const GCScanner& mr_scanner;
		GCSingleScatterSimulator m_sss;
		const GCHistogram3D* mp_promptsHis;
		const GCHistogram3D* mp_randomsHis;
		const GCHistogram3D* mp_normOrSensHis;
		const GCHistogram3D* mp_acfHis;

		std::vector<bool> m_scatterTailsMask;
		bool m_doTailFitting;
		bool m_isNorm;
		bool m_saveIntermediary;
		float m_maskThreshold;
		size_t m_scatterTailsMaskWidth;

		std::unique_ptr<GCHistogram3DOwned> mp_scatterHisto;  // Final structure
	};
}  // namespace Scatter
