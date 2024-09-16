/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/Image.hpp"
#include "datastruct/projection/GCHistogram3D.hpp"
#include "datastruct/scanner/GCScanner.hpp"
#include "geometry/GCCylinder.hpp"
#include "geometry/GCPlane.hpp"

#include "omp.h"

class GCSingleScatterSimulation
{
public:
	GCSingleScatterSimulation(GCScanner* p_scanner, Image* p_lambda,
	                          Image* p_mu, GCHistogram3D* p_prompts_histo,
	                          GCHistogram3D* p_norm_histo,
	                          GCHistogram3D* p_acf_histo,
	                          const std::string& mu_det_file, int seedi = 13,
	                          bool p_doTailFitting = true);

	void readMuDetFile(const std::string& mu_det_file);
	void run_SSS(size_t numberZ, size_t numberPhi, size_t numberR,
	             bool printProgress = false);
	double compute_single_scatter_in_lor(GCStraightLineParam* lor);

	GCHistogram3DOwned* getScatterHistogram() { return mp_scatterHisto.get(); }

protected:
	double ran1(int* idum);
	double get_mu_scaling_factor(double energy);
	double get_klein_nishina(double cosa);
	double get_intersection_length_lor_crystal(GCStraightLineParam* lor);
	bool pass_collimator(GCStraightLineParam* lor);
	double get_mu_det(double energy);

public:
	int nsamples;
	std::vector<double> xsamp, ysamp, zsamp;                // mu image samples
	std::vector<size_t> samples_z, samples_phi, samples_r;  // Histogram samples
	float energy_lld, sigma_energy;
	float rdet, thickdet, afovdet, rcoll;

protected:
	GCScanner* mp_scanner;
	GCHistogram3D* mp_promptsHisto;
	GCHistogram3D* mp_normHisto;
	GCHistogram3D* mp_acfHisto;
	Image* mp_mu;      // Attenuation image
	Image* mp_lambda;  // Image from one iteration
	GCCylinder m_cyl1, m_cyl2;
	GCPlane m_endPlate1, m_endPlate2;

	std::unique_ptr<double[]> mp_muDetTable;
	bool m_doTailFitting;
	const float m_maskThreshold = 1.05;

	std::unique_ptr<GCHistogram3DOwned> mp_scatterHisto;  // Final structure
};
