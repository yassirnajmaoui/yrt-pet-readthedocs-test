/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "datastruct/image/GCImageDevice.cuh"

#include "datastruct/image/GCImage.hpp"
#include "datastruct/image/GCImageSpaceKernels.cuh"
#include "operators/GCOperatorDevice.cuh"
#include "utils/GCAssert.hpp"
#include "utils/GCGPUMemory.cuh"
#include "utils/GCGPUTypes.cuh"
#include "utils/GCGPUUtils.cuh"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

void py_setup_gcimagedevice(py::module& m)
{
	auto c = py::class_<GCImageDevice, GCImageBase>(m, "GCImageDevice");
	c.def(
	    "TransferToDeviceMemory",
	    [](GCImageDevice& self, py::buffer& np_data)
	    {
		    try
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
			    const std::vector dims = {self.getParams().nz,
			                              self.getParams().ny,
			                              self.getParams().nx};
			    for (int i = 0; i < 3; i++)
			    {
				    if (buffer.shape[i] != dims[i])
				    {
					    throw std::invalid_argument(
					        "The buffer shape does not match with the image "
					        "parameters");
				    }
			    }
			    self.transferToDeviceMemory(
			        reinterpret_cast<float*>(buffer.ptr), true);
		    }
		    catch (const std::exception& e)
		    {
			    std::cerr << "Error in given buffer: " << e.what() << std::endl;
		    }
	    },
	    "Copy from numpy array to device", "numpy_array"_a);
	c.def(
	    "TransferToDeviceMemory",
	    [](GCImageDevice& self, const GCImage* sourceImage)
	    { self.transferToDeviceMemory(sourceImage, true); },
	    "Copy from a Host-side GCImage object", "sourceImage"_a);
	c.def(
	    "TransferToHostMemory",
	    [](const GCImageDevice& self)
	    {
		    const auto& params = self.getParams();
		    const size_t elemStride = sizeof(float);

		    const size_t size = params.nz * params.ny * params.nx;
		    float* float_ptr = new float[size];
		    self.transferToHostMemory(float_ptr, true);

		    // Create a Python object that will free the allocated
		    // memory when destroyed:
		    py::capsule free_when_done(
		        float_ptr,
		        [](void* f)
		        {
			        auto* float_ptr = reinterpret_cast<float*>(f);
			        std::cerr << "Element [0] = " << float_ptr[0] << "\n";
			        std::cerr << "freeing memory @ " << f << "\n";
			        delete[] float_ptr;
		        });

		    return py::array_t<float, py::array::c_style>(
		        {params.nz, params.ny, params.nx},  // dimensions
		        {elemStride * params.nx * params.ny, elemStride * params.nx,
		         elemStride},     // strides
		        float_ptr,        // the data pointer
		        free_when_done);  // deleter
	    },
	    "Transfer device image to host-side memory and return it as a numpy "
	    "array");
	c.def(
	    "TransferToHostMemory", [](GCImageDevice& self, GCImage* img)
	    { self.transferToHostMemory(img); },
	    "Transfer device image to host-side GCImage", "image"_a);

	c.def("setValue", &GCImageDevice::setValue, "initValue"_a = 0.0);
	c.def("addFirstImageToSecond", &GCImageDevice::addFirstImageToSecond,
	      "imgOut"_a);
	c.def("applyThreshold", &GCImageDevice::applyThreshold, "maskImage"_a,
	      "threshold"_a, "val_le_scale"_a, "val_le_off"_a, "val_gt_scale"_a,
	      "val_gt_off"_a);
	c.def("updateEMThreshold", &GCImageDevice::updateEMThreshold, "updateImg"_a,
	      "normImg"_a, "threshold"_a);
	c.def("writeToFile", &GCImageDevice::writeToFile, "image_fname"_a);

	c.def("applyThresholdDevice", &GCImageDevice::applyThresholdDevice,
	      "maskImg"_a, "threshold"_a, "val_le_scale"_a, "val_le_off"_a,
	      "val_gt_scale"_a, "val_gt_off"_a);

	auto c_owned =
	    py::class_<GCImageDeviceOwned, GCImageDevice>(m, "GCImageDeviceOwned");
	c_owned.def(
	    py::init(
	        [](const GCImageParams& imgParams) {
		        return std::make_unique<GCImageDeviceOwned>(imgParams, nullptr);
	        }),
	    "Create GCImageDevice using image parameters (will not allocate)",
	    "img_params"_a);
	c_owned.def(
	    py::init(
	        [](const GCImageParams& imgParams, const std::string& filename) {
		        return std::make_unique<GCImageDeviceOwned>(imgParams, filename,
		                                                    nullptr);
	        }),
	    "Create GCImageDevice using image parameters and filename",
	    "img_params"_a, "filename"_a);
	c_owned.def(
	    py::init(
	        [](const GCImage* img_ptr)
	        { return std::make_unique<GCImageDeviceOwned>(img_ptr, nullptr); }),
	    "Create a GCImageDevice using a host-size GCImage", "img"_a);
	c_owned.def("allocate",
	            [](GCImageDeviceOwned& self) { self.allocate(true); });

	auto c_alias =
	    py::class_<GCImageDeviceAlias, GCImageDevice>(m, "GCImageDeviceAlias");
	c_alias.def(
	    py::init(
	        [](const GCImageParams& imgParams) {
		        return std::make_unique<GCImageDeviceAlias>(imgParams, nullptr);
	        }),
	    "Create GCImageDevice using image parameters (will not allocate)",
	    "img_params"_a);
	c_alias.def("getDevicePointer", &GCImageDeviceAlias::getDevicePointerInULL);
	c_alias.def("setDevicePointer",
	            static_cast<void (GCImageDeviceAlias::*)(size_t)>(
	                &GCImageDeviceAlias::setDevicePointer),
	            "Set a device address for the image array. For "
	            "usage with PyTorch, use \'myArray.data_ptr()\'",
	            "data_ptr"_a);
	c_alias.def("isDevicePointerSet", &GCImageDeviceAlias::isDevicePointerSet,
	            "Returns true if the device pointer is not null");
}

#endif  // if BUILD_PYBIND11


GCImageDevice::GCImageDevice(const GCImageParams& imgParams,
                             const cudaStream_t* stream_ptr)
    : GCImageBase(imgParams), mp_stream(stream_ptr)
{
	m_launchParams = Util::initiateDeviceParameters(imgParams);
	m_imgSize = imgParams.nx * imgParams.ny * imgParams.nz;
}

const cudaStream_t* GCImageDevice::getStream() const
{
	return mp_stream;
}

size_t GCImageDevice::getImageSize() const
{
	return m_imgSize;
}

void GCImageDevice::transferToDeviceMemory(const float* hp_img_ptr,
                                           bool p_synchronize)
{
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");
	Util::copyHostToDevice(getDevicePointer(), hp_img_ptr, m_imgSize, mp_stream,
	                       p_synchronize);
}

void GCImageDevice::transferToDeviceMemory(const GCImage* hp_img_ptr,
                                           bool p_synchronize)
{
	ASSERT_MSG(getParams().isSameDimensionsAs(hp_img_ptr->getParams()),
	           "Image dimensions mismatch");

	m_tempBuffer.reAllocateIfNeeded(m_imgSize);
	float* h_floatTempBuffer_ptr = m_tempBuffer.getPointer();
	const double* h_double_ptr = hp_img_ptr->getData().GetRawPointer();
	ASSERT_MSG(h_double_ptr != nullptr,
	           "Either the image given is invalid or not allocated");

	for (int id = 0; id < static_cast<int>(m_imgSize); id++)
	{
		h_floatTempBuffer_ptr[id] = static_cast<float>(h_double_ptr[id]);
	}
	transferToDeviceMemory(h_floatTempBuffer_ptr, p_synchronize);
}

void GCImageDevice::transferToHostMemory(float* hp_img_ptr,
                                         bool p_synchronize) const
{
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");
	Util::copyDeviceToHost(hp_img_ptr, getDevicePointer(), m_imgSize, mp_stream,
	                       p_synchronize);
}

void GCImageDevice::transferToHostMemory(GCImage* hp_img_ptr,
                                         bool p_synchronize) const
{
	m_tempBuffer.reAllocateIfNeeded(m_imgSize);
	float* tempBufPointer = m_tempBuffer.getPointer();
	std::cout << "Transferring image from Device to Host..." << std::endl;
	transferToHostMemory(tempBufPointer, p_synchronize);
	std::cout << "Done transferring image from Device to Host." << std::endl;

	ASSERT(hp_img_ptr != nullptr);
	// Note: Eventually we'll need to use float32 for images in order to avoid
	// caveats like this
	double* hp_double_ptr = hp_img_ptr->getData().GetRawPointer();
	ASSERT_MSG(hp_double_ptr != nullptr,
	           "The GCImage provided is not yet allocated or bound");
	for (int id = 0; id < static_cast<int>(m_imgSize); id++)
	{
		hp_double_ptr[id] = static_cast<double>(tempBufPointer[id]);
	}
}

void GCImageDevice::applyThresholdDevice(const GCImageDevice* maskImg,
                                         const float threshold,
                                         const float val_le_scale,
                                         const float val_le_off,
                                         const float val_gt_scale,
                                         const float val_gt_off)
{
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");

	if (mp_stream != nullptr)
	{
		applyThreshold_kernel<<<m_launchParams.gridSize,
		                        m_launchParams.blockSize, 0, *mp_stream>>>(
		    getDevicePointer(), maskImg->getDevicePointer(), threshold,
		    val_le_scale, val_le_off, val_gt_scale, val_gt_off, getParams().nx,
		    getParams().ny, getParams().nz);
		cudaStreamSynchronize(*mp_stream);
	}
	else
	{
		applyThreshold_kernel<<<m_launchParams.gridSize,
		                        m_launchParams.blockSize>>>(
		    getDevicePointer(), maskImg->getDevicePointer(), threshold,
		    val_le_scale, val_le_off, val_gt_scale, val_gt_off, getParams().nx,
		    getParams().ny, getParams().nz);
		cudaDeviceSynchronize();
	}
	cudaCheckError();
}

void GCImageDevice::applyThreshold(const GCImageBase* maskImg, double threshold,
                                   double val_le_scale, double val_le_off,
                                   double val_gt_scale, double val_gt_off)
{
	const auto maskImg_GCImageDevice =
	    dynamic_cast<const GCImageDevice*>(maskImg);
	ASSERT_MSG(maskImg_GCImageDevice != nullptr,
	           "Input image has the wrong type");

	applyThresholdDevice(
	    maskImg_GCImageDevice, static_cast<float>(threshold),
	    static_cast<float>(val_le_scale), static_cast<float>(val_le_off),
	    static_cast<float>(val_gt_scale), static_cast<float>(val_gt_off));
}

void GCImageDevice::writeToFile(const std::string& image_fname) const
{
	auto tmpImage = std::make_unique<GCImageOwned>(getParams());
	tmpImage->allocate();
	transferToHostMemory(tmpImage.get(), true);
	tmpImage->writeToFile(image_fname);
}

void GCImageDevice::updateEMThreshold(GCImageBase* updateImg,
                                      const GCImageBase* normImg,
                                      double threshold)
{
	auto* updateImg_GCImageDevice = dynamic_cast<GCImageDevice*>(updateImg);
	const auto* normImg_GCImageDevice =
	    dynamic_cast<const GCImageDevice*>(normImg);

	ASSERT_MSG(updateImg_GCImageDevice != nullptr,
	           "updateImg is not GCImageDevice");
	ASSERT_MSG(normImg != nullptr, "normImg is not GCImageDevice");
	ASSERT_MSG(
	    updateImg_GCImageDevice->getParams().isSameDimensionsAs(getParams()),
	    "Image dimensions mismatch");
	ASSERT_MSG(
	    normImg_GCImageDevice->getParams().isSameDimensionsAs(getParams()),
	    "Image dimensions mismatch");
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");

	if (mp_stream != nullptr)
	{
		updateEM_kernel<<<m_launchParams.gridSize, m_launchParams.blockSize, 0,
		                  *mp_stream>>>(
		    updateImg_GCImageDevice->getDevicePointer(), getDevicePointer(),
		    normImg_GCImageDevice->getDevicePointer(), getParams().nx,
		    getParams().ny, getParams().nz, threshold);
		cudaStreamSynchronize(*mp_stream);
	}
	else
	{
		updateEM_kernel<<<m_launchParams.gridSize, m_launchParams.blockSize>>>(
		    updateImg_GCImageDevice->getDevicePointer(), getDevicePointer(),
		    normImg_GCImageDevice->getDevicePointer(), getParams().nx,
		    getParams().ny, getParams().nz, threshold);
		cudaDeviceSynchronize();
	}
	cudaCheckError();
}

void GCImageDevice::addFirstImageToSecond(GCImageBase* second) const
{
	auto* second_GCImageDevice = dynamic_cast<GCImageDevice*>(second);

	ASSERT_MSG(second_GCImageDevice != nullptr, "imgOut is not GCImageDevice");
	ASSERT_MSG(
	    second_GCImageDevice->getParams().isSameDimensionsAs(getParams()),
	    "Image dimensions mismatch");
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");

	if (mp_stream != nullptr)
	{
		addFirstImageToSecond_kernel<<<
		    m_launchParams.gridSize, m_launchParams.blockSize, 0, *mp_stream>>>(
		    getDevicePointer(), second_GCImageDevice->getDevicePointer(),
		    getParams().nx, getParams().ny, getParams().nz);
		cudaStreamSynchronize(*mp_stream);
	}
	else
	{
		addFirstImageToSecond_kernel<<<m_launchParams.gridSize,
		                               m_launchParams.blockSize>>>(
		    getDevicePointer(), second_GCImageDevice->getDevicePointer(),
		    getParams().nx, getParams().ny, getParams().nz);
		cudaDeviceSynchronize();
	}
	cudaCheckError();
}

void GCImageDevice::setValue(double initValue)
{
	ASSERT_MSG(getDevicePointer() != nullptr, "Device Image not allocated yet");

	if (mp_stream != nullptr)
	{
		setValue_kernel<<<m_launchParams.gridSize, m_launchParams.blockSize, 0,
		                  *mp_stream>>>(
		    getDevicePointer(), static_cast<float>(initValue), getParams().nx,
		    getParams().ny, getParams().nz);
		cudaStreamSynchronize(*mp_stream);
	}
	else
	{
		setValue_kernel<<<m_launchParams.gridSize, m_launchParams.blockSize>>>(
		    getDevicePointer(), static_cast<float>(initValue), getParams().nx,
		    getParams().ny, getParams().nz);
		cudaDeviceSynchronize();
	}
	cudaCheckError();
}


GCImageDeviceOwned::GCImageDeviceOwned(const GCImageParams& imgParams,
                                       const cudaStream_t* stream_ptr)
    : GCImageDevice(imgParams, stream_ptr), mpd_devicePointer(nullptr)
{
}

GCImageDeviceOwned::GCImageDeviceOwned(const GCImageParams& imgParams,
                                       const std::string& filename,
                                       const cudaStream_t* stream_ptr)
    : GCImageDevice(imgParams, stream_ptr), mpd_devicePointer(nullptr)
{
	readFromFile(filename);
}

GCImageDeviceOwned::GCImageDeviceOwned(const GCImage* img_ptr,
                                       const cudaStream_t* stream_ptr)
    : GCImageDevice(img_ptr->getParams(), stream_ptr),
      mpd_devicePointer(nullptr)
{
	allocate(false);
	transferToDeviceMemory(img_ptr);
}

GCImageDeviceOwned::~GCImageDeviceOwned()
{
	if (mpd_devicePointer != nullptr)
	{
		std::cout << "Freeing image device buffer..." << std::endl;
		Util::deallocateDevice(mpd_devicePointer);
	}
}

void GCImageDeviceOwned::allocate(bool synchronize)
{
	const auto& params = getParams();
	std::cout << "Allocating device memory for an image of dimensions "
	          << "[" << params.nz << ", " << params.ny << ", " << params.nx
	          << "]..." << std::endl;

	Util::allocateDevice(&mpd_devicePointer, m_imgSize, mp_stream, false);
	Util::memsetDevice(mpd_devicePointer, 0, m_imgSize, mp_stream, synchronize);

	std::cout << "Done allocating device memory." << std::endl;
}

void GCImageDeviceOwned::readFromFile(const std::string& filename)
{
	// Create temporary GCImage
	const auto img = std::make_unique<GCImageOwned>(getParams(), filename);
	allocate(false);
	transferToDeviceMemory(img.get(), true);
}

float* GCImageDeviceOwned::getDevicePointer()
{
	return mpd_devicePointer;
}

const float* GCImageDeviceOwned::getDevicePointer() const
{
	return mpd_devicePointer;
}


GCImageDeviceAlias::GCImageDeviceAlias(const GCImageParams& imgParams,
                                       const cudaStream_t* stream_ptr)
    : GCImageDevice(imgParams, stream_ptr), mpd_devicePointer(nullptr)
{
}

float* GCImageDeviceAlias::getDevicePointer()
{
	return mpd_devicePointer;
}

const float* GCImageDeviceAlias::getDevicePointer() const
{
	return mpd_devicePointer;
}

size_t GCImageDeviceAlias::getDevicePointerInULL() const
{
	return reinterpret_cast<size_t>(mpd_devicePointer);
}

void GCImageDeviceAlias::setDevicePointer(float* ppd_devicePointer)
{
	mpd_devicePointer = ppd_devicePointer;
}

void GCImageDeviceAlias::setDevicePointer(size_t ppd_pointerInULL)
{
	setDevicePointer(reinterpret_cast<float*>(ppd_pointerInULL));
}

bool GCImageDeviceAlias::isDevicePointerSet() const
{
	return mpd_devicePointer != nullptr;
}
