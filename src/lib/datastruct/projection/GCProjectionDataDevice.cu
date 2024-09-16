/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/projection/GCProjectionDataDevice.cuh"
#include "datastruct/projection/GCProjectionSpaceKernels.cuh"
#include "datastruct/projection/GCUniformHistogram.hpp"
#include "datastruct/projection/IListMode.hpp"
#include "operators/GCOperatorDevice.cuh"
#include "utils/GCAssert.hpp"
#include "utils/GCGlobals.hpp"

#include "omp.h"
#include <utility>

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

void py_setup_gcprojectiondatadevice(py::module& m)
{
	auto c = py::class_<GCProjectionDataDevice, GCProjectionList>(
	    m, "GCProjectionDataDevice");
	c.def(
	    "loadEventLORs",
	    [](GCProjectionDataDevice& self, size_t subsetId, size_t batchId,
	       const ImageParams& imgParams)
	    { self.loadEventLORs(subsetId, batchId, imgParams); },
	    "Load the LORs of a specific batch in a specific subset", "subsetId"_a,
	    "batchId"_a, "imgParams"_a);
	c.def("transferProjValuesToHost",
	      [](GCProjectionDataDevice& self, IProjectionData* dest)
	      { self.transferProjValuesToHost(dest); });
	c.def("loadProjValuesFromHost",
	      [](GCProjectionDataDevice& self, const IProjectionData* src)
	      { self.loadProjValuesFromHost(src, nullptr); });
	c.def("loadProjValuesFromReference", [](GCProjectionDataDeviceOwned& self)
	      { self.loadProjValuesFromReference(); });
	c.def("getCurrentBatchSize", &GCProjectionDataDevice::getCurrentBatchSize);
	c.def("getCurrentBatchId", &GCProjectionDataDevice::getCurrentBatchId);
	c.def("getCurrentSubsetId", &GCProjectionDataDevice::getCurrentSubsetId);
	c.def("getNumBatches", &GCProjectionDataDevice::getNumBatches);
	c.def("areLORsGathered", &GCProjectionDataDevice::areLORsGathered);

	auto c_owned =
	    py::class_<GCProjectionDataDeviceOwned, GCProjectionDataDevice>(
	        m, "GCProjectionDataDeviceOwned");
	c_owned.def(
	    py::init<const GCScanner*, const IProjectionData*, int, float>(),
	    "Create a GCProjectionDataDevice. This constructor will also store "
	    "the Scanner LUT in the device",
	    "scanner"_a, "reference"_a, "num_OSEM_subsets"_a = 1,
	    "shareOfMemoryToUse"_a = GCProjectionDataDevice::DefaultMemoryShare);
	c_owned.def(
	    py::init<const GCProjectionDataDevice*>(),
	    "Create a GCProjectionDataDevice from an existing one. They will "
	    "share the LORs",
	    "orig"_a);
	c_owned.def("allocateForProjValues", [](GCProjectionDataDeviceOwned& self)
	            { self.allocateForProjValues(); });

	auto c_alias =
	    py::class_<GCProjectionDataDeviceAlias, GCProjectionDataDevice>(
	        m, "GCProjectionDataDeviceAlias");
	c_alias.def(
	    py::init<const GCScanner*, const IProjectionData*, int, float>(),
	    "Create a GCProjectionDataDeviceAlias. This constructor will also "
	    "store "
	    "the Scanner LUT in the device",
	    "scanner"_a, "reference"_a, "num_OSEM_subsets"_a = 1,
	    "shareOfMemoryToUse"_a = GCProjectionDataDevice::DefaultMemoryShare);
	c_alias.def(
	    py::init<const GCProjectionDataDevice*>(),
	    "Create a GCProjectionDataDeviceAlias from an existing one. They will "
	    "share the LORs",
	    "orig"_a);
	c_alias.def("getProjValuesDevicePointer",
	            &GCProjectionDataDeviceAlias::getProjValuesDevicePointerInULL);
	c_alias.def("setProjValuesDevicePointer",
	            static_cast<void (GCProjectionDataDeviceAlias::*)(size_t)>(
	                &GCProjectionDataDeviceAlias::setProjValuesDevicePointer),
	            "Set a device address for the projection values array. For "
	            "usage with PyTorch, use \'myArray.data_ptr()\'",
	            "data_ptr"_a);
	c_alias.def("isDevicePointerSet",
	            &GCProjectionDataDeviceAlias::isDevicePointerSet,
	            "Returns true if the device pointer is not null");
}

#endif  // if BUILD_PYBIND11

GCProjectionDataDevice::GCProjectionDataDevice(
    const GCProjectionDataDevice* orig)
    : GCProjectionList(orig->mp_reference),
      mp_binIteratorList(orig->mp_binIteratorList),
      mp_scanner(orig->mp_scanner),
      mp_LORs(orig->mp_LORs)
{
	for (const auto& origBatchSetup : orig->m_batchSetups)
	{
		m_batchSetups.push_back(origBatchSetup);
	}
}

GCProjectionDataDevice::GCProjectionDataDevice(
    const GCScanner* pp_scanner, const IProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    float shareOfMemoryToUse)
    : GCProjectionList(pp_reference),
      mp_binIteratorList(std::move(pp_binIteratorList)),
      mp_scanner(pp_scanner)
{
	mp_LORs = std::make_unique<GCLORsDevice>(mp_scanner);
	createBatchSetups(shareOfMemoryToUse);
}

GCProjectionDataDevice::GCProjectionDataDevice(
    std::shared_ptr<GCLORsDevice> pp_LORs, const IProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    float shareOfMemoryToUse)
    : GCProjectionList(pp_reference),
      mp_binIteratorList(std::move(pp_binIteratorList)),
      mp_LORs(std::move(pp_LORs))
{
	mp_scanner = mp_LORs->getScanner();
	createBatchSetups(shareOfMemoryToUse);
}

GCProjectionDataDevice::GCProjectionDataDevice(
    std::shared_ptr<GCLORsDevice> pp_LORs, const IProjectionData* pp_reference,
    int num_OSEM_subsets, float shareOfMemoryToUse)
    : GCProjectionList(pp_reference), mp_LORs(std::move(pp_LORs))
{
	mp_scanner = mp_LORs->getScanner();
	createBinIterators(num_OSEM_subsets);
	createBatchSetups(shareOfMemoryToUse);
}

GCProjectionDataDevice::GCProjectionDataDevice(
    std::shared_ptr<GCScannerDevice> pp_scannerDevice,
    const IProjectionData* pp_reference, int num_OSEM_subsets,
    float shareOfMemoryToUse)
    : GCProjectionList(pp_reference), mp_scanner(pp_scannerDevice->getScanner())
{
	createBinIterators(num_OSEM_subsets);
	createBatchSetups(shareOfMemoryToUse);
	mp_LORs = std::make_unique<GCLORsDevice>(std::move(pp_scannerDevice));
}

GCProjectionDataDevice::GCProjectionDataDevice(
    const GCScanner* pp_scanner, const IProjectionData* pp_reference,
    int num_OSEM_subsets, float shareOfMemoryToUse)
    : GCProjectionList(pp_reference), mp_scanner(pp_scanner)
{
	createBinIterators(num_OSEM_subsets);
	createBatchSetups(shareOfMemoryToUse);

	mp_LORs = std::make_unique<GCLORsDevice>(mp_scanner);
}

void GCProjectionDataDevice::createBinIterators(int num_OSEM_subsets)
{
	m_binIterators.reserve(num_OSEM_subsets);
	for (int subsetId = 0; subsetId < num_OSEM_subsets; subsetId++)
	{
		m_binIterators.push_back(
		    mp_reference->getBinIter(num_OSEM_subsets, subsetId));
		mp_binIteratorList.push_back(m_binIterators.at(subsetId).get());
	}
}

void GCProjectionDataDevice::createBatchSetups(float shareOfMemoryToUse)
{
	size_t memAvailable = getDeviceInfo(true);
	// Shrink memory according to the portion we want to use
	memAvailable = static_cast<size_t>(static_cast<float>(memAvailable) *
	                                   shareOfMemoryToUse);

	const size_t possibleEventsPerBatch =
	    memAvailable /
	    (GCGlobalsCuda::threadsPerBlockData * MemoryUsagePerEvent) *
	    GCGlobalsCuda::threadsPerBlockData;

	const size_t numSubsets = mp_binIteratorList.size();
	m_batchSetups.reserve(numSubsets);
	for (size_t subsetId = 0; subsetId < numSubsets; subsetId++)
	{
		m_batchSetups.emplace_back(mp_binIteratorList.at(subsetId)->size(),
		                           possibleEventsPerBatch);
	}
}

void GCProjectionDataDevice::loadEventLORs(size_t subsetId, size_t batchId,
                                           const ImageParams& imgParams,
                                           const cudaStream_t* stream)
{
	mp_LORs->loadEventLORs(*mp_binIteratorList.at(subsetId),
	                       m_batchSetups.at(subsetId), subsetId, batchId,
	                       *mp_reference, imgParams, stream);
}

void GCProjectionDataDevice::loadProjValuesFromReference(
    const cudaStream_t* stream)
{
	loadProjValuesFromHostInternal(getReference(), nullptr, stream);
}

void GCProjectionDataDevice::loadProjValuesFromHost(const IProjectionData* src,
                                                    const cudaStream_t* stream)
{
	loadProjValuesFromHostInternal(src, nullptr, stream);
}

void GCProjectionDataDevice::loadProjValuesFromHost(const IProjectionData* src,
                                                    const IHistogram* histo,
                                                    const cudaStream_t* stream)
{
	loadProjValuesFromHostInternal(src, histo, stream);
}

void GCProjectionDataDevice::loadProjValuesFromHostInternal(
    const IProjectionData* src, const IHistogram* histo,
    const cudaStream_t* stream)
{
	if (src->isUniform() && histo == nullptr)
	{
		// No need to "getProjectionValue" everywhere, just fill the buffer with
		// the same value
		clearProjections(getReference()->getProjectionValue(0), stream);
	}
	else
	{
		const size_t batchSize = getCurrentBatchSize();
		ASSERT_MSG(batchSize > 0,
		           "The Batch size is 0. You didn't load the LORs "
		           "before loading the projection values");

		m_tempBuffer.reAllocateIfNeeded(batchSize);
		float* projValuesBuffer = m_tempBuffer.getPointer();

		auto* binIter = mp_binIteratorList.at(getCurrentSubsetId());
		const size_t firstBatchSize =
		    getBatchSetup(getCurrentSubsetId()).getBatchSize(0);
		const size_t offset = getCurrentBatchId() * firstBatchSize;

		size_t binIdx;
		bin_t binId;
		if (histo == nullptr)
		{
			// Fill the buffer using the source directly
#pragma omp parallel for default(none) private(binIdx, binId) \
    firstprivate(offset, binIter, projValuesBuffer, src, batchSize)
			for (binIdx = 0; binIdx < batchSize; binIdx++)
			{
				binId = binIter->get(binIdx + offset);
				projValuesBuffer[binIdx] = src->getProjectionValue(binId);
			}
		}
		else
		{
			// Fill the buffer using the corresponding value in the histogram
			histo_bin_t histoBin;
#pragma omp parallel for default(none) private(binIdx, binId, histoBin) \
    firstprivate(offset, binIter, projValuesBuffer, src, batchSize, histo)
			for (binIdx = 0; binIdx < batchSize; binIdx++)
			{
				binId = binIter->get(binIdx + offset);
				histoBin = src->getHistogramBin(binId);
				projValuesBuffer[binIdx] =
				    histo->getProjectionValueFromHistogramBin(histoBin);
			}
		}

		Util::copyHostToDevice(getProjValuesDevicePointer(), projValuesBuffer,
		                       batchSize, stream, true);
	}
}

void GCProjectionDataDevice::transferProjValuesToHost(
    IProjectionData* projDataDest, const cudaStream_t* stream) const
{
	const size_t batchSize = getCurrentBatchSize();
	ASSERT_MSG(batchSize > 0, "The Batch size is 0. You didn't load the LORs "
	                          "before loading the projection values");

	m_tempBuffer.reAllocateIfNeeded(batchSize);
	float* projValuesBuffer = m_tempBuffer.getPointer();
	Util::copyDeviceToHost(projValuesBuffer, getProjValuesDevicePointer(),
	                       batchSize, stream, true);

	auto* binIter = mp_binIteratorList.at(getCurrentSubsetId());
	const size_t firstBatchSize =
	    m_batchSetups.at(getCurrentSubsetId()).getBatchSize(0);
	const size_t offset = getCurrentBatchId() * firstBatchSize;

	size_t binIdx;
	bin_t binId;
#pragma omp parallel for default(none) private(binIdx, binId) \
    firstprivate(offset, binIter, projValuesBuffer, projDataDest, batchSize)
	for (binIdx = 0; binIdx < batchSize; binIdx++)
	{
		binId = binIter->get(binIdx + offset);
		projDataDest->setProjectionValue(binId, projValuesBuffer[binIdx]);
	}
}

std::shared_ptr<GCScannerDevice>
    GCProjectionDataDevice::getScannerDevice() const
{
	return mp_LORs->getScannerDevice();
}

const GCScanner* GCProjectionDataDevice::getScanner() const
{
	return mp_scanner;
}

size_t GCProjectionDataDevice::getCurrentBatchSize() const
{
	return mp_LORs->getLoadedBatchSize();
}

size_t GCProjectionDataDevice::getCurrentBatchId() const
{
	return mp_LORs->getLoadedBatchId();
}

size_t GCProjectionDataDevice::getCurrentSubsetId() const
{
	return mp_LORs->getLoadedSubsetId();
}

const float4* GCProjectionDataDevice::getLorDet1PosDevicePointer() const
{
	return mp_LORs->getLorDet1PosDevicePointer();
}

const float4* GCProjectionDataDevice::getLorDet1OrientDevicePointer() const
{
	return mp_LORs->getLorDet1OrientDevicePointer();
}

const float4* GCProjectionDataDevice::getLorDet2PosDevicePointer() const
{
	return mp_LORs->getLorDet2PosDevicePointer();
}

const float4* GCProjectionDataDevice::getLorDet2OrientDevicePointer() const
{
	return mp_LORs->getLorDet2OrientDevicePointer();
}

const float* GCProjectionDataDevice::getLorTOFValueDevicePointer() const
{
	return mp_LORs->getLorTOFValueDevicePointer();
}

float GCProjectionDataDevice::getProjectionValue(bin_t id) const
{
	(void)id;
	throw std::logic_error("Disabled function in Device-side class");
}

void GCProjectionDataDevice::setProjectionValue(bin_t id, float val)
{
	(void)id;
	(void)val;
	throw std::logic_error("Disabled function in Device-side class");
}

void GCProjectionDataDevice::clearProjections(float value)
{
	clearProjections(value, nullptr);
}

void GCProjectionDataDevice::clearProjections(float value,
                                              const cudaStream_t* stream)
{
	if (value == 0.0f)
	{
		clearProjections(stream);
		return;
	}
	const size_t batchSize = getCurrentBatchSize();
	const auto launchParams = Util::initiateDeviceParameters(batchSize);

	if (stream != nullptr)
	{
		clearProjections_kernel<<<launchParams.gridSize, launchParams.blockSize,
		                          0, *stream>>>(
		    getProjValuesDevicePointer(), value, static_cast<int>(batchSize));
		cudaStreamSynchronize(*stream);
	}
	else
	{
		clearProjections_kernel<<<launchParams.gridSize,
		                          launchParams.blockSize>>>(
		    getProjValuesDevicePointer(), value, static_cast<int>(batchSize));
		cudaDeviceSynchronize();
	}
	cudaCheckError();
}

void GCProjectionDataDevice::clearProjections(const cudaStream_t* stream)
{
	if (stream != nullptr)
	{
		cudaMemsetAsync(getProjValuesDevicePointer(), 0,
		                sizeof(float) * getCurrentBatchSize(), *stream);
		cudaStreamSynchronize(*stream);
	}
	else
	{
		cudaMemset(getProjValuesDevicePointer(), 0,
		           sizeof(float) * getCurrentBatchSize());
		cudaDeviceSynchronize();
	}
	cudaCheckError();
}

void GCProjectionDataDevice::divideMeasurements(
    const IProjectionData* measurements, const BinIterator* binIter)
{
	divideMeasurements(measurements, binIter, nullptr);
}

void GCProjectionDataDevice::divideMeasurements(
    const IProjectionData* measurements, const BinIterator* binIter,
    const cudaStream_t* stream)
{
	(void)binIter;  // Not needed as this class has its own BinIterators
	const auto* measurements_device =
	    dynamic_cast<const GCProjectionDataDevice*>(measurements);
	const size_t batchSize = getCurrentBatchSize();
	const auto launchParams = Util::initiateDeviceParameters(batchSize);

	if (stream != nullptr)
	{
		divideMeasurements_kernel<<<launchParams.gridSize,
		                            launchParams.blockSize, 0, *stream>>>(
		    measurements_device->getProjValuesDevicePointer(),
		    getProjValuesDevicePointer(), static_cast<int>(batchSize));
		cudaStreamSynchronize(*stream);
	}
	else
	{
		divideMeasurements_kernel<<<launchParams.gridSize,
		                            launchParams.blockSize>>>(
		    measurements_device->getProjValuesDevicePointer(),
		    getProjValuesDevicePointer(), static_cast<int>(batchSize));
		cudaDeviceSynchronize();
	}
	cudaCheckError();
}

void GCProjectionDataDevice::addProjValues(
    const GCProjectionDataDevice* projValues, const cudaStream_t* stream)
{
	const size_t batchSize = getCurrentBatchSize();
	const auto launchParams = Util::initiateDeviceParameters(batchSize);

	if (stream != nullptr)
	{
		addProjValues_kernel<<<launchParams.gridSize, launchParams.blockSize, 0,
		                       *stream>>>(
		    projValues->getProjValuesDevicePointer(),
		    getProjValuesDevicePointer(), static_cast<int>(batchSize));
		cudaStreamSynchronize(*stream);
	}
	else
	{
		addProjValues_kernel<<<launchParams.gridSize, launchParams.blockSize>>>(
		    projValues->getProjValuesDevicePointer(),
		    getProjValuesDevicePointer(), static_cast<int>(batchSize));
		cudaDeviceSynchronize();
	}
	cudaCheckError();
}

const GCGPUBatchSetup&
    GCProjectionDataDevice::getBatchSetup(size_t subsetId) const
{
	return m_batchSetups.at(subsetId);
}

size_t GCProjectionDataDevice::getNumBatches(size_t subsetId) const
{
	return m_batchSetups.at(subsetId).getNumBatches();
}

bool GCProjectionDataDevice::areLORsGathered() const
{
	return mp_LORs->areLORsGathered();
}

GCProjectionDataDeviceOwned::GCProjectionDataDeviceOwned(
    const GCScanner* pp_scanner, const IProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    float shareOfMemoryToUse)
    : GCProjectionDataDevice(pp_scanner, pp_reference,
                             std::move(pp_binIteratorList), shareOfMemoryToUse)
{
	mp_projValues = std::make_unique<GCDeviceArray<float>>();
}

GCProjectionDataDeviceOwned::GCProjectionDataDeviceOwned(
    const GCScanner* pp_scanner, const IProjectionData* pp_reference,
    int num_OSEM_subsets, float shareOfMemoryToUse)
    : GCProjectionDataDevice(pp_scanner, pp_reference, num_OSEM_subsets,
                             shareOfMemoryToUse)
{
	mp_projValues = std::make_unique<GCDeviceArray<float>>();
}

GCProjectionDataDeviceOwned::GCProjectionDataDeviceOwned(
    std::shared_ptr<GCScannerDevice> pp_scannerDevice,
    const IProjectionData* pp_reference, int num_OSEM_subsets,
    float shareOfMemoryToUse)
    : GCProjectionDataDevice(std::move(pp_scannerDevice), pp_reference,
                             num_OSEM_subsets, shareOfMemoryToUse)
{
	mp_projValues = std::make_unique<GCDeviceArray<float>>();
}

GCProjectionDataDeviceOwned::GCProjectionDataDeviceOwned(
    std::shared_ptr<GCLORsDevice> pp_LORs, const IProjectionData* pp_reference,
    int num_OSEM_subsets, float shareOfMemoryToUse)
    : GCProjectionDataDevice(std::move(pp_LORs), pp_reference, num_OSEM_subsets,
                             shareOfMemoryToUse)
{
	mp_projValues = std::make_unique<GCDeviceArray<float>>();
}

GCProjectionDataDeviceOwned::GCProjectionDataDeviceOwned(
    std::shared_ptr<GCLORsDevice> pp_LORs, const IProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    float shareOfMemoryToUse)
    : GCProjectionDataDevice(std::move(pp_LORs), pp_reference,
                             std::move(pp_binIteratorList), shareOfMemoryToUse)
{
	mp_projValues = std::make_unique<GCDeviceArray<float>>();
}

GCProjectionDataDeviceOwned::GCProjectionDataDeviceOwned(
    const GCProjectionDataDevice* orig)
    : GCProjectionDataDevice(orig)
{
	mp_projValues = std::make_unique<GCDeviceArray<float>>();
}

float* GCProjectionDataDeviceOwned::getProjValuesDevicePointer()
{
	return mp_projValues->getDevicePointer();
}

const float* GCProjectionDataDeviceOwned::getProjValuesDevicePointer() const
{
	return mp_projValues->getDevicePointer();
}

void GCProjectionDataDeviceOwned::allocateForProjValues(
    const cudaStream_t* stream)
{
	mp_projValues->allocate(getCurrentBatchSize(), stream);
}

void GCProjectionDataDeviceOwned::loadProjValuesFromHostInternal(
    const IProjectionData* src, const IHistogram* histo,
    const cudaStream_t* stream)
{
	if (!mp_projValues->isAllocated())
	{
		allocateForProjValues(stream);
	}
	GCProjectionDataDevice::loadProjValuesFromHostInternal(src, histo, stream);
}

GCProjectionDataDeviceAlias::GCProjectionDataDeviceAlias(
    const GCScanner* pp_scanner, const IProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    float shareOfMemoryToUse)
    : GCProjectionDataDevice(pp_scanner, pp_reference,
                             std::move(pp_binIteratorList), shareOfMemoryToUse),
      mpd_devicePointer(nullptr)
{
}

GCProjectionDataDeviceAlias::GCProjectionDataDeviceAlias(
    const GCScanner* pp_scanner, const IProjectionData* pp_reference,
    int num_OSEM_subsets, float shareOfMemoryToUse)
    : GCProjectionDataDevice(pp_scanner, pp_reference, num_OSEM_subsets,
                             shareOfMemoryToUse),
      mpd_devicePointer(nullptr)
{
}

GCProjectionDataDeviceAlias::GCProjectionDataDeviceAlias(
    std::shared_ptr<GCScannerDevice> pp_scannerDevice,
    const IProjectionData* pp_reference, int num_OSEM_subsets,
    float shareOfMemoryToUse)
    : GCProjectionDataDevice(std::move(pp_scannerDevice), pp_reference,
                             num_OSEM_subsets, shareOfMemoryToUse),
      mpd_devicePointer(nullptr)
{
}

GCProjectionDataDeviceAlias::GCProjectionDataDeviceAlias(
    std::shared_ptr<GCLORsDevice> pp_LORs, const IProjectionData* pp_reference,
    int num_OSEM_subsets, float shareOfMemoryToUse)
    : GCProjectionDataDevice(std::move(pp_LORs), pp_reference, num_OSEM_subsets,
                             shareOfMemoryToUse),
      mpd_devicePointer(nullptr)
{
}

GCProjectionDataDeviceAlias::GCProjectionDataDeviceAlias(
    std::shared_ptr<GCLORsDevice> pp_LORs, const IProjectionData* pp_reference,
    std::vector<const BinIterator*> pp_binIteratorList,
    float shareOfMemoryToUse)
    : GCProjectionDataDevice(std::move(pp_LORs), pp_reference,
                             std::move(pp_binIteratorList), shareOfMemoryToUse),
      mpd_devicePointer(nullptr)
{
}

GCProjectionDataDeviceAlias::GCProjectionDataDeviceAlias(
    const GCProjectionDataDevice* orig)
    : GCProjectionDataDevice(orig), mpd_devicePointer(nullptr)
{
}

float* GCProjectionDataDeviceAlias::getProjValuesDevicePointer()
{
	return mpd_devicePointer;
}

const float* GCProjectionDataDeviceAlias::getProjValuesDevicePointer() const
{
	return mpd_devicePointer;
}

size_t GCProjectionDataDeviceAlias::getProjValuesDevicePointerInULL() const
{
	return reinterpret_cast<size_t>(mpd_devicePointer);
}

void GCProjectionDataDeviceAlias::setProjValuesDevicePointer(
    float* ppd_devicePointer)
{
	mpd_devicePointer = ppd_devicePointer;
}

void GCProjectionDataDeviceAlias::setProjValuesDevicePointer(
    size_t ppd_pointerInULL)
{
	mpd_devicePointer = reinterpret_cast<float*>(ppd_pointerInULL);
}

bool GCProjectionDataDeviceAlias::isDevicePointerSet() const
{
	return mpd_devicePointer != nullptr;
}
