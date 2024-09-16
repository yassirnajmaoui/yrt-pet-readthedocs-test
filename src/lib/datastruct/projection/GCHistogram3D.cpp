/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/GCHistogram3D.hpp"

#include "geometry/GCConstants.hpp"

GCHistogram3D::GCHistogram3D(const GCScanner* pp_scanner)
    : mp_data(nullptr), mp_scanner(pp_scanner)
{
	r_cut = mp_scanner->min_ang_diff / 2;
	num_doi_poss = mp_scanner->num_doi * mp_scanner->num_doi;

	n_r = num_doi_poss *
	      (mp_scanner->dets_per_ring / 2 + 1 - mp_scanner->min_ang_diff);

	n_phi = mp_scanner->dets_per_ring;

	size_t dz_max = mp_scanner->max_ring_diff;
	n_z_bin =
	    (dz_max + 1) * mp_scanner->num_rings - (dz_max * (dz_max + 1)) / 2;
	// Number of z_bins that have z1 < z2
	n_z_bin_diff = n_z_bin - mp_scanner->num_rings;
	// Other side for if z1 > z2
	n_z_bin = n_z_bin + n_z_bin_diff;
	setupHistogram();
	histoSize = n_z_bin * n_phi * n_r;
}

GCHistogram3D::~GCHistogram3D() {}

GCHistogram3DOwned::GCHistogram3DOwned(const GCScanner* pp_scanner)
    : GCHistogram3D(pp_scanner)
{
	mp_data = std::make_unique<Array3D<float>>();
}

GCHistogram3DOwned::GCHistogram3DOwned(const GCScanner* p_scanner,
                                       const std::string& filename)
    : GCHistogram3DOwned(p_scanner)
{
	readFromFile(filename);
}

GCHistogram3DAlias::GCHistogram3DAlias(const GCScanner* p_scanner)
    : GCHistogram3D(p_scanner)
{
	mp_data = std::make_unique<Array3DAlias<float>>();
}

void GCHistogram3DOwned::readFromFile(const std::string& filename)
{
	std::array<size_t, 3> dims{n_z_bin, n_phi, n_r};
	try
	{
		mp_data->readFromFile(filename, dims);
	}
	catch (const std::exception& e)
	{
		throw std::runtime_error(
		    "Error during Histogram initialization either the scanner\'s "
		    "attributes do not match the histogram given, the file given is "
		    "inexistant or the file given is not a valid histogram file");
	}
}

void GCHistogram3DAlias::Bind(Array3DBase<float>& p_data)
{
	static_cast<Array3DAlias<float>*>(mp_data.get())->bind(p_data);
	if (mp_data->getRawPointer() != p_data.getRawPointer())
	{
		throw std::runtime_error(
		    "Error occured in the binding of the given array");
	}
}

void GCHistogram3DOwned::allocate()
{
	static_cast<Array3D<float>*>(mp_data.get())->allocate(n_z_bin, n_phi, n_r);
}

void GCHistogram3D::writeToFile(const std::string& filename) const
{
	mp_data->writeToFile(filename);
}

bin_t GCHistogram3D::getBinIdFromCoords(coord_t r, coord_t phi,
                                        coord_t z_bin) const
{
	return z_bin * n_phi * n_r + phi * n_r + r;
}

void GCHistogram3D::getCoordsFromBinId(bin_t binId, coord_t& r, coord_t& phi,
                                       coord_t& z_bin) const
{
	z_bin = binId / (n_phi * n_r);
	phi = (binId % (n_phi * n_r)) / n_r;
	r = (binId % (n_phi * n_r)) % n_r;
}

size_t GCHistogram3D::count() const
{
	return histoSize;
}

void GCHistogram3D::setupHistogram()
{
	for (coord_t phi = 0; phi < n_phi; phi++)
	{
		for (coord_t r_ring = 0; r_ring < (n_r / num_doi_poss); r_ring++)
		{
			det_id_t d1_ring, d2_ring;
			getDetPairInSameRing(r_ring, phi, d1_ring, d2_ring);
			m_ringMap[{d1_ring, d2_ring}] = {r_ring, phi};
		}
	}
}

void GCHistogram3D::getDetPairFromCoords(coord_t r, coord_t phi, coord_t z_bin,
                                         det_id_t& d1, det_id_t& d2) const
{
	// Bound checks for the input r, phi, z_bin
	if (r >= n_r || phi >= n_phi || z_bin >= n_z_bin)
	{
		throw std::range_error(
		    "The requested histogram coordinates [" + std::to_string(z_bin) +
		    ", " + std::to_string(phi) + ", " + std::to_string(r) +
		    "] are outside the histogram array (size:[" +
		    std::to_string(n_z_bin) + ", " + std::to_string(n_phi) + ", " +
		    std::to_string(n_r) + "])");
	}

	coord_t r_ring = r / num_doi_poss;
	coord_t d1_ring, d2_ring;
	getDetPairInSameRing(r_ring, phi, d1_ring, d2_ring);

	coord_t doi_case = r % num_doi_poss;
	coord_t doi_d1 = doi_case % mp_scanner->num_doi;
	coord_t doi_d2 = doi_case / mp_scanner->num_doi;

	// Determine delta Z and Z1
	coord_t z1, z2;
	if (z_bin < mp_scanner->num_rings)
	{
		z1 = z2 = z_bin;
	}
	else
	{
		// Regardless of if z1<z2 or z1>z2
		int current_z_bin =
		    (((int)z_bin) - mp_scanner->num_rings) % n_z_bin_diff +
		    mp_scanner->num_rings;
		int current_n_planes = mp_scanner->num_rings;
		size_t delta_z = 0;
		while (current_z_bin - current_n_planes >= 0)
		{
			current_z_bin -= current_n_planes;
			current_n_planes -= 1;
			delta_z++;
		}
		z1 = current_z_bin;
		z2 = z1 + delta_z;
		// Check if i had to switch (z1>z2)
		if (z_bin - mp_scanner->num_rings >= n_z_bin_diff)
		{
			std::swap(z1, z2);
		}
	}

	d1 = d1_ring + z1 * mp_scanner->dets_per_ring +
	     doi_d1 * (mp_scanner->dets_per_ring * mp_scanner->num_rings);
	d2 = d2_ring + z2 * mp_scanner->dets_per_ring +
	     doi_d2 * (mp_scanner->dets_per_ring * mp_scanner->num_rings);
}

// Transpose
void GCHistogram3D::getCoordsFromDetPair(det_id_t d1, det_id_t d2, coord_t& r,
                                         coord_t& phi, coord_t& z_bin) const
{
	coord_t r_ring;
	if (d1 > d2)
		std::swap(d1, d2);
	det_id_t d1_ring = d1 % (mp_scanner->dets_per_ring);
	det_id_t d2_ring = d2 % (mp_scanner->dets_per_ring);
	if (d1_ring > d2_ring)
	{
		std::swap(d1, d2);
		std::swap(d1_ring, d2_ring);
	}

	getCoordsInSameRing(d1_ring, d2_ring, r_ring, phi);
	det_id_t doi_d1 = d1 / (mp_scanner->num_rings * mp_scanner->dets_per_ring);
	det_id_t doi_d2 = d2 / (mp_scanner->num_rings * mp_scanner->dets_per_ring);
	if (d1_ring > d2_ring)
		std::swap(doi_d1, doi_d2);
	r = r_ring * num_doi_poss + (doi_d1 + doi_d2 * mp_scanner->num_doi);

	int z1 = (d1 / (mp_scanner->dets_per_ring)) % (mp_scanner->num_rings);
	int z2 = (d2 / (mp_scanner->dets_per_ring)) % (mp_scanner->num_rings);

	coord_t delta_z = static_cast<coord_t>(std::abs(z2 - z1));
	coord_t num_removed_z_bins = delta_z * (delta_z - 1) / 2;
	z_bin =
	    delta_z * mp_scanner->num_rings + std::min(z1, z2) - num_removed_z_bins;
	if (delta_z > 0 && z1 > z2)
	{
		z_bin += n_z_bin_diff;  // switch
	}
	if (z_bin >= n_z_bin)
	{
		throw std::range_error("The detector pair given does not respect the "
		                       "maximum ring difference rule");
	}
}

det_pair_t GCHistogram3D::getDetPairFromBinId(bin_t binId) const
{
	coord_t r, phi, z_bin;
	det_id_t d1, d2;
	getCoordsFromBinId(binId, r, phi, z_bin);
	getDetPairFromCoords(r, phi, z_bin, d1, d2);
	return {d1, d2};
}

bin_t GCHistogram3D::getBinIdFromDetPair(det_id_t d1, det_id_t d2) const
{
	coord_t r, phi, z_bin;
	getCoordsFromDetPair(d1, d2, r, phi, z_bin);
	return getBinIdFromCoords(r, phi, z_bin);
}

histo_bin_t GCHistogram3D::getHistogramBin(bin_t bin) const
{
	return bin;
}

void GCHistogram3D::getDetPairInSameRing(coord_t r_ring, coord_t phi,
                                         det_id_t& d1_ring,
                                         det_id_t& d2_ring) const
{
	int n_tot_ring = mp_scanner->dets_per_ring;
	int r = r_ring;  // for cleanness
	int d01 = 0;
	int d02 = n_tot_ring / 2;
	if (phi % 2 != 0)
		d02 = n_tot_ring / 2 + 1;
	int dr1 = d01 + (r - n_tot_ring / 4 + r_cut);
	int dr2 = d02 - (r - n_tot_ring / 4 + r_cut);
	if (dr1 < 0)
		dr1 += n_tot_ring;
	if (dr2 < 0)
		dr2 += n_tot_ring;
	int d1_ring_i = (dr1 + phi / 2) % n_tot_ring;
	int d2_ring_i = (dr2 + phi / 2) % n_tot_ring;
	d1_ring = static_cast<det_id_t>(d1_ring_i);
	d2_ring = static_cast<det_id_t>(d2_ring_i);
	if (d1_ring > d2_ring)
		std::swap(d1_ring, d2_ring);
}

bool GCHistogram3D::getCoordsInSameRing_safe(det_id_t d1_ring, det_id_t d2_ring,
                                             coord_t& r_ring,
                                             coord_t& phi) const
{
	det_id_t d1_ring_c = d1_ring;
	det_id_t d2_ring_c = d2_ring;
	if (d1_ring_c > d2_ring_c)
		std::swap(d1_ring_c, d2_ring_c);

	auto got = m_ringMap.find({d1_ring_c, d2_ring_c});
	if (got == m_ringMap.end())
	{
		std::cerr << "Detector ring Exception found at " << d1_ring_c << ", "
		          << d2_ring_c << std::endl;
		return false;
	}
	DetRingCoordinates coords = got->second;
	r_ring = coords[0];
	phi = coords[1];
	return true;
}

void GCHistogram3D::getCoordsInSameRing(det_id_t d1_ring, det_id_t d2_ring,
                                        coord_t& r_ring, coord_t& phi) const
{
	// Note: ".at(...)" is the "const" version of "[...]"
	size_t d1_ring_c = d1_ring;
	size_t d2_ring_c = d2_ring;
	if (d1_ring_c > d2_ring_c)
		std::swap(d1_ring_c, d2_ring_c);
	try
	{
		DetRingCoordinates coords = m_ringMap.at({d1_ring_c, d2_ring_c});
		r_ring = coords[0];
		phi = coords[1];
	}
	catch (const std::out_of_range& e)
	{
		std::cerr << "Caught: " << e.what() << std::endl;
		throw std::range_error(
		    "A Line-Of-Response might not respect the Minimum-Angle-Difference "
		    "restriction of the scanner.\nCheck the scanner properties or the "
		    "input files.");
	}
}

float GCHistogram3D::getProjectionValue(bin_t binId) const
{
	return mp_data->get_flat(binId);
}

void GCHistogram3D::setProjectionValue(bin_t binId, float val)
{
	mp_data->set_flat(binId, val);
}

void GCHistogram3D::incrementProjection(bin_t binId, float val)
{
	mp_data->increment_flat(binId, val);
}

det_id_t GCHistogram3D::getDetector1(bin_t evId) const
{
	auto [d1, _] = getDetPairFromBinId(evId);
	return d1;
}

det_id_t GCHistogram3D::getDetector2(bin_t evId) const
{
	auto [_, d2] = getDetPairFromBinId(evId);
	return d2;
}

det_pair_t GCHistogram3D::getDetectorPair(bin_t id) const
{
	return getDetPairFromBinId(id);
}

void GCHistogram3D::get_z1_z2(coord_t z_bin, coord_t& z1, coord_t& z2) const
{
	if (z_bin < mp_scanner->num_rings)
	{
		z1 = z2 = z_bin;
	}
	else
	{
		// Regardless of if z1<z2 or z1>z2
		int current_z_bin =
		    (((int)z_bin) - mp_scanner->num_rings) % n_z_bin_diff +
		    mp_scanner->num_rings;
		int current_n_planes = mp_scanner->num_rings;
		size_t delta_z = 0;
		while (current_z_bin - current_n_planes >= 0)
		{
			current_z_bin -= current_n_planes;
			current_n_planes -= 1;
			delta_z++;
		}
		z1 = current_z_bin;
		z2 = z1 + delta_z;
		// Check if i had to switch (z1>z2)
		if (z_bin - mp_scanner->num_rings >= n_z_bin_diff)
		{
			std::swap(z1, z2);
		}
	}
}

void GCHistogram3D::clearProjections()
{
	clearProjections(0.0f);
}

void GCHistogram3D::clearProjections(float value)
{
	mp_data->fill(value);
}

std::unique_ptr<GCBinIterator> GCHistogram3D::getBinIter(int numSubsets,
                                                         int idxSubset) const
{
	if (idxSubset < 0 || numSubsets <= 0)
		throw std::invalid_argument(
		    "The subset index cannot be negative, the number of subsets cannot "
		    "be less or equal than zero");
	if (idxSubset >= numSubsets)
		throw std::invalid_argument(
		    "The subset index has to be smaller than the number of subsets");
	return std::make_unique<GCBinIteratorRangeHistogram3D>(
	    n_z_bin, n_phi, n_r, numSubsets, idxSubset);
}

float GCHistogram3D::getProjectionValueFromHistogramBin(
    histo_bin_t histoBinId) const
{
	if (std::holds_alternative<bin_t>(histoBinId))
	{
		// Use bin itself
		return getProjectionValue(std::get<bin_t>(histoBinId));
	}

	// use the detector pair
	const auto [d1, d2] = std::get<det_pair_t>(histoBinId);
	const bin_t binId = getBinIdFromDetPair(d1, d2);
	return getProjectionValue(binId);
}


#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
void py_setup_gchistogram3d(pybind11::module& m)
{
	auto c = py::class_<GCHistogram3D, IHistogram>(m, "GCHistogram3D",
	                                               py::buffer_protocol());
	c.def("getScanner", &GCHistogram3D::getScanner);
	c.def_readonly("n_z_bin", &GCHistogram3D::n_z_bin);
	c.def_readonly("n_phi", &GCHistogram3D::n_phi);
	c.def_readonly("n_r", &GCHistogram3D::n_r);
	c.def_readonly("histoSize", &GCHistogram3D::histoSize);
	c.def_buffer(
	    [](GCHistogram3D& self) -> py::buffer_info
	    {
		    Array3DBase<float>& d = self.getData();
		    return py::buffer_info(d.getRawPointer(), sizeof(float),
		                           py::format_descriptor<float>::format(), 3,
		                           d.getDims(), d.getStrides());
	    });
	c.def("writeToFile", &GCHistogram3D::writeToFile);
	c.def("getBinIdFromCoords", &GCHistogram3D::getBinIdFromCoords);
	c.def("getCoordsFromBinId",
	      [](const GCHistogram3D& self, size_t binId)
	      {
		      if (binId >= self.histoSize)
			      throw std::invalid_argument(
			          "The binId provided exceeds the size "
			          "permitted by the histogram");
		      coord_t r, phi, z_bin;
		      self.getCoordsFromBinId(binId, r, phi, z_bin);
		      return py::make_tuple(r, phi, z_bin);
	      });
	c.def(
	    "getDetPairFromCoords",
	    [](const GCHistogram3D& self, coord_t r, coord_t phi, coord_t z_bin)
	    {
		    if (r >= self.n_r)
		    {
			    throw std::invalid_argument("The r coordinate given does not "
			                                "respect the histogram shape");
		    }
		    if (phi >= self.n_phi)
		    {
			    throw std::invalid_argument("The phi coordinate given does not "
			                                "respect the histogram shape");
		    }
		    if (z_bin >= self.n_z_bin)
		    {
			    throw std::invalid_argument("The z_bin coordinate given does "
			                                "not respect the histogram shape");
		    }
		    det_id_t d1, d2;
		    self.getDetPairFromCoords(r, phi, z_bin, d1, d2);
		    return py::make_tuple(d1, d2);
	    });
	c.def("getDetPairFromBinId",
	      [](const GCHistogram3D& self, bin_t binId)
	      {
		      if (binId >= self.histoSize)
			      throw std::invalid_argument(
			          "The binId provided exceeds the size "
			          "permitted by the histogram");
		      auto [d1, d2] = self.getDetPairFromBinId(binId);
		      return py::make_tuple(d1, d2);
	      });
	c.def("getCoordsFromDetPair",
	      [](const GCHistogram3D& self, det_id_t d1, det_id_t d2)
	      {
		      coord_t r, phi, z_bin;
		      self.getCoordsFromDetPair(d1, d2, r, phi, z_bin);
		      return py::make_tuple(r, phi, z_bin);
	      });
	c.def("getBinIdFromDetPair", &GCHistogram3D::getBinIdFromDetPair);
	c.def(
	    "get_z1_z2",
	    [](const GCHistogram3D& self, coord_t z_bin)
	    {
		    if (z_bin >= self.n_z_bin)
			    throw std::invalid_argument(
			        "The binId provided exceeds the size "
			        "permitted by the histogram");
		    coord_t z1, z2;
		    self.get_z1_z2(z_bin, z1, z2);
		    return py::make_tuple(z1, z2);
	    },
	    py::arg("z_bin"));

	auto c_owned =
	    py::class_<GCHistogram3DOwned, GCHistogram3D>(m, "GCHistogram3DOwned");
	c_owned.def(py::init<GCScanner*>());
	c_owned.def(py::init<GCScanner*, std::string>());
	c_owned.def("readFromFile", &GCHistogram3DOwned::readFromFile);
	c_owned.def("allocate", &GCHistogram3DOwned::allocate);

	auto c_alias =
	    py::class_<GCHistogram3DAlias, GCHistogram3D>(m, "GCHistogram3DAlias");
	c_alias.def(py::init<GCScanner*>());
	c_alias.def("Bind", &GCHistogram3DAlias::Bind);
	c_alias.def(
	    "Bind",
	    [](GCHistogram3DAlias& self,
	       py::buffer& np_data)
	    {
		    py::buffer_info buffer = np_data.request();
		    if (buffer.ndim != 3)
		    {
			    throw std::invalid_argument(
			        "The buffer given has to have 3 dimensions");
		    }
		    if (buffer.format != py::format_descriptor<float>::format())
		    {
			    throw std::invalid_argument(
			        "The buffer given has to have a float32 format");
		    }
		    std::vector<size_t> dims = {self.n_z_bin, self.n_phi, self.n_r};
		    for (int i = 0; i < 3; i++)
		    {
			    if (buffer.shape[i] != static_cast<int>(dims[i]))
			    {
				    throw std::invalid_argument(
				        "The buffer shape does not match with the image "
				        "parameters");
			    }
		    }
		    static_cast<Array3DAlias<float>&>(self.getData())
		        .bind(reinterpret_cast<float*>(buffer.ptr), dims[0], dims[1],
		              dims[2]);
	    });
}
#endif

std::unique_ptr<IProjectionData>
    GCHistogram3DOwned::create(const GCScanner& scanner,
                               const std::string& filename,
                               const Plugin::OptionsResult& pluginOptions)
{
	(void)pluginOptions;  // No use for extra options
	return std::make_unique<GCHistogram3DOwned>(&scanner, filename);
}

Plugin::OptionsListPerPlugin GCHistogram3DOwned::getOptions()
{
	// No extra options
	return {};
}

REGISTER_PROJDATA_PLUGIN("H", GCHistogram3DOwned, GCHistogram3DOwned::create,
                         GCHistogram3DOwned::getOptions)
