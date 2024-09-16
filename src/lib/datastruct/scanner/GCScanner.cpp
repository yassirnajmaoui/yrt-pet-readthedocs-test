/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/scanner/GCScanner.hpp"

#include "datastruct/scanner/GCDetRegular.hpp"
#include "utils/GCJSONUtils.hpp"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void py_setup_gcscanner(pybind11::module& m)
{
	auto c = py::class_<GCScanner>(m, "GCScanner");
	c.def("getNumDets", &GCScanner::getNumDets);
	c.def("getTheoreticalNumDets", &GCScanner::getTheoreticalNumDets);
	c.def("getDetectorPos", &GCScanner::getDetectorPos);
	c.def("getDetectorOrient", &GCScanner::getDetectorOrient);
	c.def_readwrite("scannerName", &GCScanner::scannerName);
	c.def_readwrite("axialFOV", &GCScanner::axialFOV);
	c.def_readwrite("crystalSize_z", &GCScanner::crystalSize_z);
	c.def_readwrite("crystalSize_trans", &GCScanner::crystalSize_trans);
	c.def_readwrite("crystalDepth", &GCScanner::crystalDepth);
	c.def_readwrite("scannerRadius", &GCScanner::scannerRadius);
	c.def_readwrite("collimatorRadius", &GCScanner::collimatorRadius);
	c.def_readwrite("fwhm", &GCScanner::fwhm);
	c.def_readwrite("energyLLD", &GCScanner::energyLLD);
	c.def_readwrite("dets_per_ring", &GCScanner::dets_per_ring);
	c.def_readwrite("fwhm", &GCScanner::fwhm);
	c.def_readwrite("num_rings", &GCScanner::num_rings);
	c.def_readwrite("num_doi", &GCScanner::num_doi);
	c.def_readwrite("max_ring_diff", &GCScanner::max_ring_diff);
	c.def_readwrite("min_ang_diff", &GCScanner::min_ang_diff);
	c.def_readwrite("dets_per_block", &GCScanner::dets_per_block);
	c.def("createLUT",
	      [](const GCScanner& s)
	      {
		      auto lut = std::make_unique<Array2D<float>>();
		      s.createLUT(*lut.get());
		      size_t shape[2]{s.getNumDets(), 6};
		      auto lut_array = py::array_t<float>(
		          shape, std::move(lut.get())->GetRawPointer());
		      return lut_array;
	      });

	auto c_alias = py::class_<GCScannerAlias, GCScanner>(m, "GCScannerAlias");
	c_alias.def(py::init());
	c_alias.def("setDetectorSetup", &GCScannerAlias::setDetectorSetup);

	auto c_owned = py::class_<GCScannerOwned, GCScanner>(m, "GCScannerOwned");
	c_owned.def(py::init());
	c_owned.def(py::init<const std::string&>());
	c_owned.def("readFromFile", &GCScannerOwned::readFromFile);
	c_owned.def("readFromString", &GCScannerOwned::readFromString);
	c_owned.def_property(
	    "scannerPath",
	    [](const GCScannerOwned& self) { return self.getScannerPath(); },
	    [](GCScannerOwned& self, const std::string& str)
	    { self.setScannerPath(str); });
	c_owned.def(pybind11::pickle([](const GCScannerOwned& s)
	                             { return s.getScannerPath(); },
	                             [](const std::string& t)
	                             {
		                             std::unique_ptr<GCScannerOwned> s =
		                                 std::make_unique<GCScannerOwned>(t);
		                             return s;
	                             }));
}
#endif

GCScanner::GCScanner() : mp_detectors(nullptr) {}
GCScannerOwned::GCScannerOwned() : GCScanner() {}
GCScannerAlias::GCScannerAlias() : GCScanner() {}
GCScannerOwned::GCScannerOwned(const std::string& p_definitionFile)
    : GCScannerOwned()
{
	readFromFile(p_definitionFile);
}

size_t GCScanner::getNumDets() const
{
	return mp_detectors->getNumDets();
}

size_t GCScanner::getTheoreticalNumDets() const
{
	return num_doi * num_rings * dets_per_ring;
}

GCVector GCScanner::getDetectorPos(det_id_t id) const
{
	return mp_detectors->getPos(id);
}

GCVector GCScanner::getDetectorOrient(det_id_t id) const
{
	return mp_detectors->getOrient(id);
}

void GCScanner::createLUT(Array2D<float>& lut) const
{
	lut.allocate(this->getNumDets(), 6);
	for (size_t i = 0; i < this->getNumDets(); i++)
	{
		GCVector pos = mp_detectors->getPos(i);
		GCVector orient = mp_detectors->getOrient(i);
		lut[i][0] = pos.x;
		lut[i][1] = pos.y;
		lut[i][2] = pos.z;
		lut[i][3] = orient.x;
		lut[i][4] = orient.y;
		lut[i][5] = orient.z;
	}
}

void GCScannerOwned::readFromString(const std::string& fileContents)
{
	json j;
	try
	{
		j = json::parse(fileContents);
	}
	catch (json::exception& e)
	{
		throw std::invalid_argument("Error in Scanner JSON file parsing");
	}

	float scannerFileVersion = 0.0;
	Util::getParam<float>(&j, &scannerFileVersion, "VERSION", 0.0, true,
	                      "Missing VERSION in scanner definition file");

	Util::getParam<std::string>(
	    &j, &scannerName, "scannerName", "", true,
	    "Missing scanner name in scanner definition file");

	std::string detCoord;
	const bool isDetCoordGiven =
	    Util::getParam<std::string>(&j, &detCoord, "detCoord", "", false);

	Util::getParam<float>(
	    &j, &axialFOV, "axialFOV", 0.0, true,
	    "Missing Axial Field Of View value in scanner definition file");
	Util::getParam<float>(
	    &j, &crystalSize_trans, "crystalSize_trans", 0.0, true,
	    "Missing transaxial crystal size in scanner definition file");
	Util::getParam<float>(
	    &j, &crystalSize_z, "crystalSize_z", 0.0, true,
	    "Missing z-axis crystal size in scanner definition file");

	// Optional values for Scatter estimation only
	Util::getParam<float>(&j, &collimatorRadius, "collimatorRadius", -1.0,
	                      false);
	Util::getParam<float>(&j, &fwhm, "fwhm", -1.0, false);
	Util::getParam<float>(&j, &energyLLD, "energyLLD", -1.0, false);

	Util::getParam<float>(&j, &crystalDepth, "crystalDepth", 0.0, true,
	                      "Missing crystal depth in scanner definition file");
	Util::getParam<float>(&j, &scannerRadius, "scannerRadius", 0.0, true,
	                      "Missing scanner radius in scanner definition file");
	Util::getParam<size_t>(&j, &dets_per_ring, "dets_per_ring", 0, true);
	Util::getParam<size_t>(&j, &num_rings, "num_rings", 0, true);
	Util::getParam<size_t>(&j, &num_doi, "num_doi", 0, true);
	Util::getParam<size_t>(&j, &max_ring_diff, "max_ring_diff", 0, true);
	Util::getParam<size_t>(&j, &min_ang_diff, "min_ang_diff", 0, true);
	Util::getParam<size_t>(&j, &dets_per_block, "dets_per_block", 1, false);

	// Check for errors
	if (scannerFileVersion != GCSCANNER_FILE_VERSION)
	{
		throw std::invalid_argument(
		    "Wrong file version for Scanner JSON file, the "
		    "current version is " +
		    std::to_string(GCSCANNER_FILE_VERSION));
	}

	// Join paths for DetCoord
	if (isDetCoordGiven)
	{
		fs::path detCoord_path =
		    m_scannerPath.parent_path() / fs::path(detCoord);
		mp_detectorsPtr =
		    std::make_unique<GCDetCoordOwned>(detCoord_path.string());
		if (mp_detectorsPtr->getNumDets() != getTheoreticalNumDets())
			throw std::runtime_error(
			    "The number of detectors given by the LUT file does not match "
			    "the scanner's characteristics. Namely, (num_doi * num_rings * "
			    "dets_per_ring) does not equal the size of the LUT");
		this->mp_detectors = mp_detectorsPtr.get();
	}
	else
	{
		mp_detectorsPtr = std::make_unique<GCDetRegular>(this);
		static_cast<GCDetRegular*>(mp_detectorsPtr.get())->generateLUT();
		this->mp_detectors = mp_detectorsPtr.get();
	}
}

std::string GCScannerOwned::getScannerPath() const
{
	return m_scannerPath.string();
}

void GCScannerOwned::setScannerPath(const fs::path& p)
{
	m_scannerPath = p;
}

void GCScannerOwned::setScannerPath(const std::string& s)
{
	m_scannerPath = fs::path(s);
}

void GCScannerOwned::readFromFile(const std::string& p_definitionFile)
{
	m_scannerPath = fs::path(p_definitionFile);
	if (!(fs::exists(m_scannerPath)))
	{
		throw std::invalid_argument(
		    "The scanner definition file given does not exist");
	}

	std::ifstream i(m_scannerPath.string());
	auto ss = std::ostringstream{};
	ss << i.rdbuf();
	std::string file_contents = ss.str();
	readFromString(file_contents);
}
