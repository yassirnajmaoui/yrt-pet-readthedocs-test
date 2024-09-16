/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "scatter/GCSingleScatterSimulation.hpp"

#include "geometry/GCConstants.hpp"
#include "geometry/GCVector.hpp"
#include "operators/GCOperatorProjectorSiddon.hpp"
#include "utils/GCGlobals.hpp"
#include "utils/GCReconstructionUtils.hpp"
#include "utils/GCTools.hpp"

GCSingleScatterSimulation::GCSingleScatterSimulation(
    GCScanner* pp_scanner, Image* pp_lambda, Image* pp_mu,
    GCHistogram3D* pp_promptsHisto, GCHistogram3D* pp_normHisto,
    GCHistogram3D* pp_acfHisto, const std::string& muDetFile, int seedi,
    bool p_doTailFitting)
{
	// TODO yssrnjm: add Randoms correction
	mp_scanner = pp_scanner;
	mp_lambda = pp_lambda;
	mp_mu = pp_mu;
	const ImageParams& mu_params = mp_mu->getParams();
	mp_promptsHisto = pp_promptsHisto;
	mp_normHisto = pp_normHisto;
	mp_acfHisto = pp_acfHisto;
	m_doTailFitting = p_doTailFitting;

	energy_lld = mp_scanner->energyLLD;  // YP low level discriminatory energy

	sigma_energy =
	    (mp_scanner->fwhm) /
	    (2.0 * sqrt(2 * log(2.0)));  // YP: standard deviation of scattered
	                                 // photons energy distribution

	rdet = mp_scanner->scannerRadius;      // YP ring radius
	thickdet = mp_scanner->crystalDepth;   // YP detector thickness
	afovdet = mp_scanner->axialFOV;        // YP Axial FOV
	rcoll = mp_scanner->collimatorRadius;  // YP no need?

	GCVector c(0, 0, 0);
	// YP: creates 2 cylinders of axial extent "afov" in millimiters xs
	m_cyl1 = GCCylinder(c, afovdet, rdet);
	m_cyl2 = GCCylinder(c, afovdet, rdet + thickdet);
	// YP 3 points located in the last ring of the scanner
	GCVector p1(1.0, 0.0, -afovdet / 2), p2(0.0, 1.0, -afovdet / 2),
	    p3(0.0, 0.0, -afovdet / 2);
	m_endPlate1 =
	    GCPlane(p1, p2, p3);  // YP defines a plane according to these 3 points
	p1.z = p2.z = p3.z = afovdet / 2;
	m_endPlate2 = GCPlane(
	    p1, p2, p3);  // YP other plane located at the first ring of the scanner

	int seed = std::abs(seedi);  // YP random seed
	int init = -1;
	ran1(&init);
	nsamples = 0;

	// Generate scatter points:
	// YP coarser cubic grid of scatter points
	int nxsamp = static_cast<int>(mu_params.nx / 1.5);
	if (nxsamp < 5)
		nxsamp = 5;  // YP number of scatter points in x direction
	int nysamp = static_cast<int>(mu_params.ny / 1.5);
	if (nysamp < 5)
		nysamp = 5;
	int nzsamp = static_cast<int>(mu_params.nz / 1.5);
	if (nzsamp < 5)
		nzsamp = 5;
	std::cout << "nxsamp: " << nxsamp << std::endl;
	std::cout << "nysamp: " << nysamp << std::endl;
	std::cout << "nzsamp: " << nzsamp << std::endl;
	xsamp.reserve(nzsamp * nysamp * nxsamp);
	ysamp.reserve(nzsamp * nysamp * nxsamp);
	zsamp.reserve(nzsamp * nysamp * nxsamp);
	// YP spacing between scatter points
	double dxsamp = mu_params.length_x / (static_cast<double>(nxsamp));
	double dysamp = mu_params.length_y / (static_cast<double>(nysamp));
	double dzsamp = mu_params.length_z / (static_cast<double>(nzsamp));
	double x, y, z, x2, y2, z2;
	GCVector p;
	xsamp.clear();
	ysamp.clear();
	zsamp.clear();
	int i, j, k;
	for (k = 0; k < nzsamp; k++)
	{
		z = k / (static_cast<double>(nzsamp)) * mu_params.length_z -
		    mu_params.length_z / 2;
		for (j = 0; j < nysamp; j++)
		{
			y = j / (static_cast<double>(nysamp)) * mu_params.length_y -
			    mu_params.length_y / 2;
			for (i = 0; i < nxsamp; i++)
			{
				x = i / (static_cast<double>(nxsamp)) * mu_params.length_x -
				    mu_params.length_x / 2;
				x2 = ran1(&seed) * dxsamp + x;
				y2 = ran1(&seed) * dysamp + y;
				z2 = ran1(&seed) * dzsamp + z;
				// YP generate a random scatter poitn within its cell
				p.update(x2, y2, z2);
				if (mp_mu->nearest_neigh(p) > 0.005)
				{  // YP rejects the point if the associated att. coeff is below
				   // a certain threshold
					nsamples++;  // nsamp: number of scatter points
					xsamp.push_back(x2);
					ysamp.push_back(y2);
					zsamp.push_back(z2);
				}
			}
		}
	}
	if (nsamples < 10)
	{
		std::cout
		    << "Error: Small number of scatter points in "
		       "SingleScatterSimulation::SingleScatterSimulation(). nsamples="
		    << nsamples << "\n"
		    << std::endl;
		exit(-1);
	}

	// Initialize buffers
	mp_scatterHisto = std::make_unique<GCHistogram3DOwned>(mp_scanner);
	mp_scatterHisto->allocate();
	mp_scatterHisto->clearProjections();

	readMuDetFile(muDetFile);
}

void GCSingleScatterSimulation::run_SSS(size_t numberZ, size_t numberPhi,
                                        size_t numberR, bool printProgress)
{
	size_t num_i_z = numberZ;
	size_t num_i_phi = numberPhi;
	size_t num_i_r = numberR;

	if (num_i_phi <= 1)
		throw std::invalid_argument(
		    "The number of phis given has to be larger than 1");
	if (num_i_r <= 1)
		throw std::invalid_argument(
		    "The number of rs given has to be larger than 1");
	if (num_i_z == 0)
		throw std::invalid_argument(
		    "The number of zs given has to be larger than 0");

	size_t min_z = 0;
	size_t min_phi = 0;
	size_t min_r = 0;

	// We only fill the perpendicular bins... What does that imply for the
	// interpolation?
	size_t num_z = mp_scanner->num_rings;
	size_t num_phi = mp_scatterHisto->n_phi;
	size_t num_r = mp_scatterHisto->n_r;

	// Sampling
	double d_z = (num_z - min_z) / static_cast<double>(num_i_z - 1);
	double d_phi = (num_phi - min_phi) / static_cast<double>(num_i_phi - 1);
	double d_r = (num_r - min_r) / static_cast<double>(num_i_r - 1);
	samples_z.reserve(num_i_z);
	samples_phi.reserve(num_i_phi);
	samples_r.reserve(num_i_r);
	// Z dimension
	for (size_t i = 0; i < num_i_z; i++)
	{
		double z = static_cast<double>(min_z) + d_z * i;
		samples_z.push_back(std::min(static_cast<size_t>(z), num_z - 1));
	}
	for (size_t i = 0; i < num_i_phi; i++)
	{
		double phi = static_cast<double>(min_phi) + d_phi * i;
		samples_phi.push_back(std::min(static_cast<size_t>(phi), num_phi - 1));
	}
	for (size_t i = 0; i < num_i_r; i++)
	{
		double r = static_cast<double>(min_r) + d_r * i;
		samples_r.push_back(std::min(static_cast<size_t>(r), num_r - 1));
	}

	// Only used for printing purposes
	size_t progress_max = num_i_z * num_i_phi * num_i_r;
	size_t last_progress_print = 0;

	int num_threads = GCGlobals::get_num_threads();
#pragma omp parallel for schedule(static, 1) collapse(3) \
    num_threads(num_threads)
	for (size_t z_i = 0; z_i < num_i_z; z_i++)
	{
		for (size_t phi_i = 0; phi_i < num_i_phi; phi_i++)
		{
			for (size_t r_i = 0; r_i < num_i_r; r_i++)
			{
				if (printProgress)
				{
					int thread_num = omp_get_thread_num();
					if (thread_num == 0)
					{
						size_t percentage_interval = 5;
						// Print progress
						size_t progress =
						    (z_i * num_i_phi * num_i_r + phi_i * num_i_r + r_i);
						progress = progress * 100 / progress_max;
						if (progress - last_progress_print >
						    percentage_interval)
						{
							last_progress_print = progress;
							std::cout
							    << "Progress: " +
							           std::to_string(last_progress_print) + "%"
							    << std::endl;
						}
					}
				}

				size_t z = samples_z[z_i];
				size_t phi = samples_phi[phi_i];
				size_t r = samples_r[r_i];

				// Compute current LOR
				size_t scatterHistoBinId =
				    mp_scatterHisto->getBinIdFromCoords(r, phi, z);
				GCStraightLineParam lor = Util::getNativeLOR(
				    *mp_scanner, *mp_scatterHisto, scatterHistoBinId);

				float scatterResult =
				    static_cast<float>(compute_single_scatter_in_lor(&lor));
				if (scatterResult <= 0.0)
					continue;  // Ignore irrelevant lines?
				mp_scatterHisto->setProjectionValue(scatterHistoBinId,
				                                    scatterResult);
			}
		}
	}

	std::cout << "Scatter simulation completed, running linear interpolation "
	             "to fill gaps."
	          << std::endl;

	// Run interpolations to fill non-simulated bins
	size_t num_i_z_to_take = (num_i_z == 1) ? 1 : (num_i_z - 1);
	for (size_t z_i = 0; z_i < num_i_z_to_take; z_i++)
	{
		size_t z1 = samples_z[z_i];
		size_t z2 = (num_i_z == 1) ? samples_z[0] : samples_z[z_i + 1];
		for (size_t phi_i = 0; phi_i < num_i_phi - 1; phi_i++)
		{
			size_t phi1 = samples_phi[phi_i];
			size_t phi2 = samples_phi[phi_i + 1];
			for (size_t r_i = 0; r_i < num_i_r - 1; r_i++)
			{
				size_t r1 = samples_r[r_i];
				size_t r2 = samples_r[r_i + 1];
				Util::fillBox(mp_scatterHisto->getData(), z1, z2, phi1, phi2,
				              r1, r2);
			}
		}
	}
	std::cout << "Histogram filled in all the transaxial bins." << std::endl;

	std::cout << "Filling oblique bins..." << std::endl;
	for (coord_t z_bin_i = mp_scanner->num_rings;
	     z_bin_i < mp_scatterHisto->n_z_bin; ++z_bin_i)
	{
		coord_t z1, z2;
		mp_scatterHisto->get_z1_z2(z_bin_i, z1, z2);
		const Array3DBase<float>& scatterHistoData_c =
		    mp_scatterHisto->getData();
		Array3DBase<float>& scatterHistoData = mp_scatterHisto->getData();
		scatterHistoData[z_bin_i] += scatterHistoData_c[z1];
		scatterHistoData[z_bin_i] += scatterHistoData_c[z2];
		scatterHistoData[z_bin_i] *= 0.5;  // average
	}
	std::cout << "Done Filling oblique bins." << std::endl;

	if (m_doTailFitting)
	{
		std::cout << "Computing Tail-fit factor" << std::endl;
		float scat = 0.0f, prompt = 0.0f;
		for (bin_t binId = 0; binId < mp_scatterHisto->count(); binId++)
		{
			float acfValue = mp_acfHisto->getProjectionValue(binId);

			// Only fit outside the image
			if (-std::log(acfValue) > std::log(m_maskThreshold))
				continue;

			scat += mp_scatterHisto->getProjectionValue(binId);
			prompt += mp_promptsHisto->getProjectionValue(binId) *
			          mp_normHisto->getProjectionValue(binId);
		}
		float fac = prompt / scat;
		std::cout << "Tail-fitting factor: " << fac << std::endl;
		mp_scatterHisto->getData() *= fac;
	}
	std::cout << "Dividing by the ACF..." << std::endl;
	mp_scatterHisto->getData() /= mp_acfHisto->getData();
	std::cout << "Done." << std::endl;
}

// YP LOR in which to compute the scatter contribution
double GCSingleScatterSimulation::compute_single_scatter_in_lor(
    GCStraightLineParam* lor)
{
	GCVector n1 = GCVector(lor->point1.x, lor->point1.y, 0.);
	n1.normalize();
	GCVector n2 = GCVector(lor->point2.x, lor->point2.y, 0.);
	n2.normalize();

	int i;
	double res = 0., dist1, dist2, energy, cosa, mu_scaling_factor;
	double vatt, att_s_1_511, att_s_1, att_s_2_511, att_s_2;
	double dsigcompdomega, lamb_s_1, lamb_s_2, sig_s_1, sig_s_2;
	double eps_s_1_511, eps_s_1, eps_s_2_511, eps_s_2, fac1, fac2;
	double tmp, tmp511, delta_1, delta_2, mu_det, mu_det_511;
	GCStraightLineParam lor_1_s, lor_2_s;
	GCVector ps, p1, p2, u, v;

	p1.update(lor->point1);
	p2.update(lor->point2);

	tmp511 = (energy_lld - 511.0) / (sqrt(2.0) * sigma_energy);
	mu_det_511 = get_mu_det(511.0);

	for (i = 0; i < nsamples; i++)
	{  // for each scatter point in the image volume

		ps.update(xsamp[i], ysamp[i], zsamp[i]);

		// LOR going from scatter point "ps" to detector 1
		lor_1_s.update(p1, ps);
		// LOR going from scatter point "ps" to detector 2
		lor_2_s.update(p2, ps);

		// check that the rays S-det1 and S-det2 pass the end plates collimator
		// before going further:
		if (fabs(ps.z) > afovdet / 2 &&
		    (!pass_collimator(&lor_1_s) || !pass_collimator(&lor_2_s)))
			continue;


		u.update(ps - p1);
		dist1 = u.getNorm();
		u.x /= dist1;
		u.y /= dist1;
		u.z /= dist1;
		v.update(p2 - ps);
		dist2 = v.getNorm();
		v.x /= dist2;
		v.y /= dist2;
		v.z /= dist2;

		cosa = u.scalProd(v);
		// larger angle change -> more energy loss
		energy = 511.0 / (2.0 - cosa);
		if (energy <= energy_lld)
		{
			continue;
		}
		tmp = (energy_lld - energy) / (sqrt(2.0) * sigma_energy);
		mu_scaling_factor = get_mu_scaling_factor(energy);

		// get scatter values:
		vatt = mp_mu->nearest_neigh(ps);
		dsigcompdomega = get_klein_nishina(cosa);

		// compute I1 and I2:
		// TODO yssrnjm: Remove the "/10" once the right units are settled
		att_s_1_511 =
		    GCOperatorProjectorSiddon::singleForwardProjection(mp_mu, lor_1_s) /
		    10;

		att_s_1 = att_s_1_511 * mu_scaling_factor;
		lamb_s_1 = GCOperatorProjectorSiddon::singleForwardProjection(mp_lambda,
		                                                              lor_1_s);
		delta_1 = get_intersection_length_lor_crystal(&lor_1_s);
		if (delta_1 > 10 * thickdet)
		{
			std::cout << "Error computing propagation distance in detector in "
			             "SingleScatterSimulation::compute_single_scatter_in_"
			             "lor() (1).\n"
			          << std::endl;
			exit(-1);
		}

		// TODO yssrnjm: Remove the "/10" once the right units are settled
		// (again)
		att_s_2_511 =
		    GCOperatorProjectorSiddon::singleForwardProjection(mp_mu, lor_2_s) /
		    10;

		att_s_2 = att_s_2_511 * mu_scaling_factor;
		lamb_s_2 = GCOperatorProjectorSiddon::singleForwardProjection(mp_lambda,
		                                                              lor_2_s);
		delta_2 = get_intersection_length_lor_crystal(&lor_2_s);

		// Check that the distance between the two cylinders is not too big
		if (delta_2 > 10 * thickdet)
		{
			std::cout << "Error computing propagation distance in detector in "
			          << "SingleScatterSimulation::compute_single_scatter_in_"
			          << "lor() (2)." << std::endl
			          << std::endl;
			exit(-1);
		}

		// geometric efficiencies (n1 and n2 must be normalized unit vectors):
		sig_s_1 = fabs(n1.scalProd(u));
		sig_s_2 = fabs(n2.scalProd(v));

		// detection efficiencies (energy+spatial):
		eps_s_1_511 = eps_s_2_511 = Util::erfc(tmp511);
		eps_s_1 = eps_s_2 = Util::erfc(tmp);
		mu_det = get_mu_det(energy);
		eps_s_1_511 *= 1 - exp(-delta_1 * mu_det_511);
		eps_s_2_511 *= 1 - exp(-delta_2 * mu_det_511);
		eps_s_1 *= 1 - exp(-delta_1 * mu_det);
		eps_s_2 *= 1 - exp(-delta_2 * mu_det);

		fac1 = lamb_s_1 * exp(-att_s_1_511 - att_s_2);
		fac1 *= eps_s_1_511 * eps_s_2;
		fac2 = lamb_s_2 * exp(-att_s_1 - att_s_2_511);
		fac2 *= eps_s_2_511 * eps_s_1;

		res += vatt * dsigcompdomega * (fac1 + fac2) * sig_s_1 * sig_s_2 /
		       dist1 / dist1 / dist2 / dist2;
	}
	// divide the result by the sensitivity for trues for that LOR (don't do
	// this anymore because we use the sensitivity corrected scatter sinogram in
	// the reconstruction):
	u.update(p2 - p1);
	dist1 = u.getNorm();
	u.x /= dist1;
	u.y /= dist1;
	u.z /= dist1;
	sig_s_1 = fabs(n1.scalProd(u));
	sig_s_2 = fabs(n2.scalProd(u));
	eps_s_1_511 = eps_s_2_511 = Util::erfc(tmp511);
	GCVector mid(p1.x + p2.x, p1.y + p2.y, p1.z + p2.z);
	mid.x /= 2;
	mid.y /= 2;
	mid.z /= 2;
	lor_1_s.update(p1, mid);
	delta_1 = get_intersection_length_lor_crystal(&lor_1_s);
	lor_2_s.update(p2, mid);
	delta_2 = get_intersection_length_lor_crystal(&lor_2_s);
	eps_s_1_511 *= 1 - exp(-delta_1 * mu_det_511);
	eps_s_2_511 *= 1 - exp(-delta_2 * mu_det_511);
	res /= eps_s_1_511 * eps_s_1_511 * sig_s_1 * sig_s_2 / (dist1 * dist1);

	return res;
}

// This is the integrated KN formula up to a proportionaity constant:
double GCSingleScatterSimulation::get_mu_scaling_factor(double energy)
{
	double a = energy / 511.0;
	double res = (1 + a) / (a * a);
	res *= 2.0 * (1 + a) / (1 + 2.0 * a) - log(1 + 2.0 * a) / a;
	res += log(1 + 2 * a) / (2 * a) - (1 + 3 * a) / ((1 + 2 * a) * (1 + 2 * a));
	res /= 20.0 / 9.0 - 1.5 * log(3.0);
	return res;
}

// This is the differential KN formula up to a proportionelity constant for
// Ep=511keV.
double GCSingleScatterSimulation::get_klein_nishina(double cosa)
{
	double res = (1 + cosa * cosa) / 2;
	res /= (2 - cosa) * (2 - cosa);
	res *= 1 + (1 - cosa) * (1 - cosa) / ((2 - cosa) * (1 + cosa * cosa));
	return res;
}

// The first point of lor must be the detector, the second point must be the
// scatter point.
double GCSingleScatterSimulation::get_intersection_length_lor_crystal(
    GCStraightLineParam* lor)
{
	GCVector c(0.0, 0.0, 0.0), a1, a2, inter1, inter2;
	GCVector n1 = (lor->point1) - (lor->point2);  // direction of prop.
	// Compute entry point:
	m_cyl1.does_line_inter_cyl(lor, &a1, &a2);
	GCVector n2 = a1 - (lor->point2);
	if (n2.scalProd(n1) > 0)
		inter1.update(a1);
	else
		inter1.update(a2);
	// Compute out point:
	m_cyl2.does_line_inter_cyl(lor, &a1, &a2);
	n2 = a1 - (lor->point2);
	if (n2.scalProd(n1) > 0)
		inter2.update(a1);
	else
		inter2.update(a2);
	// Return distance of prop. in detector:
	double dist = (inter1 - inter2).getNorm();
	return dist;
}

// Return true if the line lor does not cross the end plates
// First point is detector, second point is scatter point
bool GCSingleScatterSimulation::pass_collimator(GCStraightLineParam* lor)
{
	if (rcoll < 1e-7)
		return true;
	GCVector inter;
	double r;
	if (lor->point2.z < 0)
		inter = m_endPlate1.findInterLine(lor);
	else
		inter = m_endPlate2.findInterLine(lor);
	r = std::sqrt(inter.x * inter.x + inter.y * inter.y);
	if (r < rcoll)
		return true;
	else
		return false;
}

double GCSingleScatterSimulation::get_mu_det(double energy)
{
	int e = static_cast<int>(energy) - 1;
	if (e < 0 || e >= 1000)
	{
		throw std::runtime_error("Error: energy out of range in "
		                         "SingleScatterSimulation::get_mu_det().");
	}
	return mp_muDetTable[e];
}

void GCSingleScatterSimulation::readMuDetFile(const std::string& mu_det_file)
{
	mp_muDetTable = std::make_unique<double[]>(1000);
	int i = 0;
	std::ifstream myfile(mu_det_file);
	if (myfile.is_open())
	{
		double a;
		while (i < 1000)
		{
			myfile >> a;
			mp_muDetTable[i++] = a;
		}
		myfile.close();
	}
	else
	{
		throw std::runtime_error(
		    "Error opening " + mu_det_file +
		    " for reading in SingleScatterSimulation::read_mu_det_file().");
	}
}

double GCSingleScatterSimulation::ran1(int* idum)
{
	int j, k;
	static int iy = 0;
	static int iv[NTAB];
	double temp;

	if (*idum <= 0 || !iy)
	{
		if (-(*idum) < 1)
			*idum = 1;
		else
			*idum = -(*idum);
		for (j = NTAB + 7; j >= 0; j--)
		{
			k = (*idum) / IQ;
			*idum = IA * (*idum - k * IQ) - IR * k;
			if (*idum < 0)
				*idum += IM;
			if (j < NTAB)
				iv[j] = *idum;
		}
		iy = iv[0];
	}
	k = (*idum) / IQ;
	*idum = IA * (*idum - k * IQ) - IR * k;
	if (*idum < 0)
		*idum += IM;
	j = iy / NDIV;
	iy = iv[j];
	iv[j] = *idum;
	if ((temp = AM * iy) > RNMX)
		return (RNMX);
	else
		return temp;
}
