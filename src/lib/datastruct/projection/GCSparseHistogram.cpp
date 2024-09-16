/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/GCSparseHistogram.hpp"
#include "utils/GCAssert.hpp"

#include <cstring>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace pybind11::literals;
namespace py = pybind11;

void py_setup_gcsparsehistogram(py::module& m)
{
	auto c = py::class_<GCSparseHistogram, IHistogram>(m, "GCSparseHistogram");
	c.def(py::init<const GCScanner&>(), "scanner"_a);
	c.def(py::init<const GCScanner&, const std::string&>(), "scanner"_a,
	      "filename"_a);
	c.def(py::init<const GCScanner&, const IProjectionData&,
	               const BinIterator*>(),
	      "scanner"_a, "projectionData"_a, "binIterator"_a = nullptr);
	c.def("allocate", &GCSparseHistogram::allocate, "numBins"_a);
	c.def("accumulate",
	      static_cast<void (GCSparseHistogram::*)(det_pair_t detPair,
	                                              float projValue)>(
	          &GCSparseHistogram::accumulate),
	      "detPair"_a, "projValue"_a);
	c.def("accumulate",
	      static_cast<void (GCSparseHistogram::*)(
	          const IProjectionData& projData, const BinIterator* binIter)>(
	          &GCSparseHistogram::accumulate),
	      "projData"_a, "binIter"_a = nullptr);
	c.def("getProjectionValueFromDetPair",
	      &GCSparseHistogram::getProjectionValueFromDetPair, "detPair"_a);
	c.def("readFromFile", &GCSparseHistogram::readFromFile, "filename"_a);
	c.def("writeToFile", &GCSparseHistogram::writeToFile, "filename"_a);
	c.def("getProjValuesArray",
	      [](GCSparseHistogram& self) -> pybind11::array_t<float>
	      {
		      const auto buf_info =
		          py::buffer_info(self.getProjectionValuesBuffer(), sizeof(float),
		                          py::format_descriptor<float>::format(), 1,
		                          {self.count()}, {sizeof(float)});
		      return py::array_t<float>(buf_info);
	      });
}
#endif

GCSparseHistogram::GCSparseHistogram(const GCScanner& pr_scanner)
    : mr_scanner(pr_scanner)
{
}

GCSparseHistogram::GCSparseHistogram(const GCScanner& pr_scanner,
                                     const std::string& filename)
    : GCSparseHistogram(pr_scanner)
{
	readFromFile(filename);
}

GCSparseHistogram::GCSparseHistogram(const GCScanner& pr_scanner,
                                     const IProjectionData& pr_projData,
                                     const BinIterator* pp_binIter)
    : GCSparseHistogram(pr_scanner)
{
	accumulate(pr_projData, pp_binIter);
}

void GCSparseHistogram::allocate(size_t numBins)
{
	m_detectorMap.reserve(numBins);
	m_projValues.reserve(numBins);
	m_detPairs.reserve(numBins);
}

template <bool IgnoreZeros>
void GCSparseHistogram::accumulate(const IProjectionData& projData,
                                   const BinIterator* binIter)
{
	size_t numBins;
	if (binIter == nullptr)
	{
		numBins = projData.count();
	}
	else
	{
		numBins = binIter->size();
	}

	allocate(numBins);
	for (bin_t bin = 0; bin < numBins; bin++)
	{
		bin_t binId = bin;
		if (binIter != nullptr)
		{
			binId = binIter->get(bin);
		}

		const float projValue = projData.getProjectionValue(binId);
		if constexpr (IgnoreZeros)
		{
			if (projValue == 0.0f)
			{
				continue;
			}
		}
		const det_pair_t detPair = projData.getDetectorPair(binId);
		// Add to the histogram
		accumulate(detPair, projValue);
	}
}
template void
    GCSparseHistogram::accumulate<true>(const IProjectionData& projData,
                                        const BinIterator* binIter);
template void
    GCSparseHistogram::accumulate<false>(const IProjectionData& projData,
                                         const BinIterator* binIter);

void GCSparseHistogram::accumulate(det_pair_t detPair, float projValue)
{
	det_pair_t newPair = SwapDetectorPairIfNeeded(detPair);

	const auto detectorMapLocation = m_detectorMap.find(newPair);
	if (detectorMapLocation != m_detectorMap.end())
	{
		// Get the proper bin
		const bin_t bin = detectorMapLocation->second;
		// Accumulate the value
		m_projValues[bin] += projValue;
	}
	else
	{
		// Create new element
		m_detectorMap.emplace(newPair, m_projValues.size());

		// Add the pair
		m_detPairs.push_back(newPair);
		// Initialize its value
		m_projValues.push_back(projValue);
	}
}

float GCSparseHistogram::getProjectionValueFromDetPair(det_pair_t detPair) const
{
	const auto detectorMapLocation =
	    m_detectorMap.find(SwapDetectorPairIfNeeded(detPair));
	if (detectorMapLocation != m_detectorMap.end())
	{
		// Get the proper bin
		const bin_t bin = detectorMapLocation->second;
		// Return the value
		return m_projValues[bin];
	}
	return 0.0f;
}

size_t GCSparseHistogram::count() const
{
	return m_projValues.size();
}

det_id_t GCSparseHistogram::getDetector1(bin_t id) const
{
	return m_detPairs[id].d1;
}

det_id_t GCSparseHistogram::getDetector2(bin_t id) const
{
	return m_detPairs[id].d2;
}

det_pair_t GCSparseHistogram::getDetectorPair(bin_t id) const
{
	return m_detPairs[id];
}

std::unique_ptr<BinIterator>
    GCSparseHistogram::getBinIter(int numSubsets, int idxSubset) const
{
	ASSERT_MSG(idxSubset < numSubsets,
	           "The subset index has to be smaller than the number of subsets");
	ASSERT_MSG(idxSubset == 0 && numSubsets == 1,
	           "Multiple subsets are not supported in sparse histograms");

	return std::make_unique<BinIteratorChronological>(numSubsets, count(),
	                                                    idxSubset);
}

float GCSparseHistogram::getProjectionValue(bin_t id) const
{
	return m_projValues[id];
}

void GCSparseHistogram::setProjectionValue(bin_t id, float val)
{
	m_projValues[id] = val;
}

float GCSparseHistogram::getProjectionValueFromHistogramBin(
    histo_bin_t histoBinId) const
{
	ASSERT(std::holds_alternative<det_pair_t>(histoBinId));
	const auto detPair = std::get<det_pair_t>(histoBinId);
	return getProjectionValueFromDetPair(detPair);
}

void GCSparseHistogram::writeToFile(const std::string& filename) const
{
	std::ofstream ofs{filename.c_str(), std::ios::binary | std::ios::out};

	if (!ofs.good())
	{
		throw std::runtime_error("Error opening file " + filename +
		                         "GCListModeLUTOwned::writeToFile.");
	}

	constexpr std::streamsize sizeOfAnEvent_bytes =
	    sizeof(det_pair_t) + sizeof(float);

	constexpr std::streamsize bufferSize_fields = (1ll << 30);
	auto buff = std::make_unique<float[]>(bufferSize_fields);
	constexpr int numFieldsPerEvent = 3;
	static_assert(numFieldsPerEvent * sizeof(float) == sizeOfAnEvent_bytes);
	const std::streamsize numEvents = count();

	std::streamoff posStart_events = 0;
	while (posStart_events < numEvents)
	{
		const std::streamsize writeSize_fields =
		    std::min(bufferSize_fields,
		             numFieldsPerEvent * (numEvents - posStart_events));
		const std::streamsize writeSize_events =
		    writeSize_fields / numFieldsPerEvent;

		for (std::streamoff i = 0; i < writeSize_events; i++)
		{
			std::memcpy(&buff[numFieldsPerEvent * i + 0], &m_detPairs[i].d1,
			            sizeof(det_id_t));
			std::memcpy(&buff[numFieldsPerEvent * i + 1], &m_detPairs[i].d2,
			            sizeof(det_id_t));
			buff[numFieldsPerEvent * i + 2] = m_projValues[i];
		}

		ofs.write(reinterpret_cast<char*>(buff.get()),
		          sizeOfAnEvent_bytes * writeSize_events);

		posStart_events += writeSize_events;
	}

	ofs.close();
}

void GCSparseHistogram::readFromFile(const std::string& filename)
{
	std::ifstream ifs{filename, std::ios::in | std::ios::binary};

	if (!ifs.good())
	{
		throw std::runtime_error("Error reading input file " + filename +
		                         "GCListModeLUTOwned::readFromFile.");
	}

	constexpr std::streamsize sizeOfAnEvent_bytes =
	    sizeof(det_pair_t) + sizeof(float);

	// Check that file has a proper size:
	ifs.seekg(0, std::ios::end);
	const std::streamsize end = ifs.tellg();
	ifs.seekg(0, std::ios::beg);
	const std::streamsize begin = ifs.tellg();
	const std::streamsize fileSize_bytes = end - begin;
	if (fileSize_bytes <= 0 || (fileSize_bytes % sizeOfAnEvent_bytes) != 0)
	{
		throw std::runtime_error("Error: Input file has incorrect size in "
		                         "GCSparseHistogram::readFromFile.");
	}
	// Compute the number of events using its size
	const std::streamsize numEvents = fileSize_bytes / sizeOfAnEvent_bytes;

	// Allocate the memory
	allocate(numEvents);

	// Prepare buffer of 3 4-byte fields
	constexpr std::streamsize bufferSize_fields = (1ll << 30);
	auto buff = std::make_unique<float[]>(bufferSize_fields);
	constexpr int numFieldsPerEvent = 3;
	static_assert(numFieldsPerEvent * sizeof(float) == sizeOfAnEvent_bytes);

	std::streamoff posStart_events = 0;
	while (posStart_events < numEvents)
	{
		const std::streamsize readSize_fields =
		    std::min(bufferSize_fields,
		             numFieldsPerEvent * (numEvents - posStart_events));
		const std::streamsize readSize_events =
		    readSize_fields / numFieldsPerEvent;

		ifs.read(reinterpret_cast<char*>(buff.get()),
		         sizeOfAnEvent_bytes * readSize_events);

		for (std::streamoff i = 0; i < readSize_events; i++)
		{
			const det_id_t d1 =
			    *reinterpret_cast<det_id_t*>(&buff[numFieldsPerEvent * i + 0]);
			const det_id_t d2 =
			    *reinterpret_cast<det_id_t*>(&buff[numFieldsPerEvent * i + 1]);
			const float projValue = buff[numFieldsPerEvent * i + 2];
			accumulate({d1, d2}, projValue);
		}

		posStart_events += readSize_events;
	}
	ifs.close();
}

float* GCSparseHistogram::getProjectionValuesBuffer()
{
	return m_projValues.data();
}

det_pair_t* GCSparseHistogram::getDetectorPairBuffer()
{
	return m_detPairs.data();
}

const float* GCSparseHistogram::getProjectionValuesBuffer() const
{
	return m_projValues.data();
}

const det_pair_t* GCSparseHistogram::getDetectorPairBuffer() const
{
	return m_detPairs.data();
}

det_pair_t GCSparseHistogram::SwapDetectorPairIfNeeded(det_pair_t detPair)
{
	auto [d1, d2] = std::minmax({detPair.d1, detPair.d2});
	return det_pair_t{d1, d2};
}

std::unique_ptr<IProjectionData>
    GCSparseHistogram::create(const GCScanner& scanner,
                              const std::string& filename,
                              const Plugin::OptionsResult& pluginOptions)
{
	(void)pluginOptions;
	return std::make_unique<GCSparseHistogram>(scanner, filename);
}

Plugin::OptionsListPerPlugin GCSparseHistogram::getOptions()
{
	return {};
}

REGISTER_PROJDATA_PLUGIN("SH", GCSparseHistogram, GCSparseHistogram::create,
                         GCSparseHistogram::getOptions)
