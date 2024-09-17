/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/GCCylinder.hpp"
#include "geometry/GCPlane.hpp"
#include "scatter/GCCrystal.hpp"

class Histogram3D;
class Scanner;
class Image;

namespace Scatter
{
	class GCSingleScatterSimulator
	{
	public:
		GCSingleScatterSimulator(const Scanner& pr_scanner,
		                         const Image& pr_mu, const Image& pr_lambda,
		                         CrystalMaterial p_crystalMaterial, int seedi);

		void runSSS(size_t numberZ, size_t numberPhi, size_t numberR,
		            Histogram3D& scatterHisto, bool printProgress = false);

		double computeSingleScatterInLOR(const GCStraightLineParam& lor) const;

		GCVector getSamplePoint(int i) const;
		int getNumSamples() const;
		bool passCollimator(const GCStraightLineParam& lor) const;

	private:
		static double Ran1(int* idum);
		static double GetKleinNishina(double cosa);
		static double GetMuScalingFactor(double energy);

		double getIntersectionLengthLORCrystal(
		    const GCStraightLineParam& lor) const;

		// Attenuation image samples
		int m_numSamples;
		std::vector<double> m_xSamples, m_ySamples, m_zSamples;
		// Histogram samples
		std::vector<size_t> m_zBinSamples, m_phiSamples, m_rSamples;

		float m_energyLLD, m_sigmaEnergy;
		float m_scannerRadius, m_crystalDepth, m_axialFOV, m_collimatorRadius;
		const Scanner& mr_scanner;
		const Image& mr_mu;      // Attenuation image
		const Image& mr_lambda;  // Image from 2 MLEM iterations
		CrystalMaterial m_crystalMaterial;
		GCCylinder m_cyl1, m_cyl2;
		GCPlane m_endPlate1, m_endPlate2;
	};
}  // namespace Scatter
