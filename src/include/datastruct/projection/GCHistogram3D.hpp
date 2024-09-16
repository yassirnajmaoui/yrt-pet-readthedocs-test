/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/PluginFramework.hpp"
#include "datastruct/projection/IHistogram.hpp"
#include "datastruct/scanner/GCScanner.hpp"
#include "utils/Array.hpp"

#include <array>
#include <vector>

// A hash function used to hash a pair of any kind
struct hash_pair
{
	template <class T1, class T2>
	int operator()(const std::pair<T1, T2>& p) const
	{
		auto hash1 = std::hash<T1>{}(p.first);
		auto hash2 = std::hash<T2>{}(p.second);
		return hash1 ^ hash2;
	}
};

typedef uint32_t coord_t;
typedef std::array<coord_t, 3> DetCoordinates;      // r, phi, z
typedef std::array<coord_t, 2> DetRingCoordinates;  // r, phi
typedef std::pair<det_id_t, det_id_t> DetRingPair;  // d1, d2

class GCHistogram3D : public IHistogram
{
public:
	const GCScanner* getScanner() const { return mp_scanner; }
	Array3DBase<float>& getData() { return *mp_data; }
	const Array3DBase<float>& getData() const { return *mp_data; }
	virtual void writeToFile(const std::string& filename) const;
	~GCHistogram3D() override = 0;

	// binId
	bin_t getBinIdFromCoords(coord_t r, coord_t phi, coord_t z_bin) const;
	void getCoordsFromBinId(bin_t binId, coord_t& r, coord_t& phi,
	                        coord_t& z_bin) const;

	// Where all the magic happens
	void getDetPairFromCoords(coord_t r, coord_t phi, coord_t z_bin,
	                          det_id_t& d1, det_id_t& d2) const;
	det_pair_t getDetPairFromBinId(bin_t binId) const;  // Uses latter
	void getCoordsFromDetPair(det_id_t d1, det_id_t d2, coord_t& r,
	                          coord_t& phi, coord_t& z_bin) const;
	bin_t getBinIdFromDetPair(det_id_t d1, det_id_t d2) const;  // Uses latter
	histo_bin_t getHistogramBin(bin_t bin) const override;

	// Functions needed for ProjectionData object and used by the operators
	float getProjectionValue(bin_t binId) const override;
	size_t count() const override;
	void setProjectionValue(bin_t binId, float val) override;
	virtual void incrementProjection(bin_t binId, float incVal);
	det_id_t getDetector1(bin_t id) const override;
	det_id_t getDetector2(bin_t id) const override;
	det_pair_t getDetectorPair(bin_t id) const override;
	void clearProjections();
	void clearProjections(float value) override;
	std::unique_ptr<GCBinIterator> getBinIter(int numSubsets,
	                                          int idxSubset) const override;

	float getProjectionValueFromHistogramBin(
	    histo_bin_t histoBinId) const override;
	void get_z1_z2(coord_t z_bin, coord_t& z1, coord_t& z2) const;

protected:
	GCHistogram3D(const GCScanner* pp_scanner);
	void getDetPairInSameRing(coord_t r_ring, coord_t phi, det_id_t& d1,
	                          det_id_t& d2) const;
	void getCoordsInSameRing(det_id_t d1_ring, det_id_t d2_ring,
	                         coord_t& r_ring, coord_t& phi) const;
	bool getCoordsInSameRing_safe(det_id_t d1_ring, det_id_t d2_ring,
	                              coord_t& r_ring, coord_t& phi) const;
	// Sets up the LUT for reverse compute
	void setupHistogram();

public:
	size_t n_r, n_phi, n_z_bin;
	size_t histoSize;
	// domains:
	// 	r: {0, 1, ..., nr/2-mad-2}
	// 	phi: {0, 1, ..., nr-1}
	// 	z: {0, 1, ..., (mrd+1)*nr-mrd*(mrd+1)/2 - 1}
	// 	where:
	// 		nr: number of detectors in a ring
	// 		mad: minimum angle difference
	// 		mrd: maximum ring difference

protected:
	std::unique_ptr<Array3DBase<float>> mp_data;
	std::unordered_map<DetRingPair, DetRingCoordinates, hash_pair> m_ringMap;
	const GCScanner* mp_scanner;
	size_t r_cut;
	size_t num_doi_poss;  // Number of DOI possibilities (ex: 2 doi -> 4 lor
	                      // possibilities)
	size_t n_z_bin_diff;  // Number of z_bins that have z1 < z2
};

class GCHistogram3DAlias : public GCHistogram3D
{
public:
	GCHistogram3DAlias(const GCScanner* p_scanner);
	void Bind(Array3DBase<float>& p_data);
};

class GCHistogram3DOwned : public GCHistogram3D
{
public:
	GCHistogram3DOwned(const GCScanner* p_scanner);
	GCHistogram3DOwned(const GCScanner* p_scanner, const std::string& filename);
	void allocate();
	void readFromFile(const std::string& filename);

	// For registering the plugin
	static std::unique_ptr<IProjectionData>
	    create(const GCScanner& scanner, const std::string& filename,
	           const Plugin::OptionsResult& pluginOptions);
	static Plugin::OptionsListPerPlugin getOptions();
};
