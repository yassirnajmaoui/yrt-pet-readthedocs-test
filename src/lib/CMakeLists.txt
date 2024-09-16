project(lib)

# List source files explicitly
set(SOURCES
        datastruct/projection/IHistogram.cpp
        datastruct/projection/GCHistogram3D.cpp
        datastruct/projection/GCUniformHistogram.cpp
        datastruct/projection/GCListModeLUT.cpp
        datastruct/projection/GCListModeLUTDOI.cpp
        datastruct/projection/GCLORMotion.cpp
        datastruct/projection/GCBinIterator.cpp
        datastruct/projection/GCProjectionList.cpp
        datastruct/projection/GCSparseHistogram.cpp
        datastruct/projection/IProjectionData.cpp
        datastruct/projection/IListMode.cpp
        datastruct/scanner/GCDetCoord.cpp
        datastruct/scanner/GCDetRegular.cpp
        datastruct/scanner/GCDetectorSetup.cpp
        datastruct/scanner/GCScanner.cpp
        datastruct/image/ImageBase.cpp
        datastruct/image/Image.cpp
        datastruct/IO.cpp
        geometry/GCStraightLineParam.cpp
        geometry/GCMatrix.cpp
        geometry/GCVector.cpp
        geometry/GCMultiRayGenerator.cpp
        geometry/GCTubeOfResponse.cpp
        geometry/GCCylinder.cpp
        geometry/GCPlane.cpp
        kernel/GCKernel.cpp
        kinetic/srtm.cpp
        motion/ImageWarperMatrix.cpp
        motion/ImageWarperTemplate.cpp
        motion/ImageWarperFunction.cpp
        operators/GCOperatorProjector.cpp
        operators/GCOperatorProjectorSiddon.cpp
        operators/GCOperatorProjectorDD.cpp
        operators/GCOperatorPsf.cpp
        operators/GCProjectionPsfManager.cpp
        operators/GCTimeOfFlight.cpp
        recon/GCOSEM.cpp
        recon/GCOSEM_cpu.cpp
        scatter/GCScatterEstimator.cpp
        scatter/GCSingleScatterSimulator.cpp
        scatter/GCCrystal.cpp
        utils/GCAssert.cpp
        utils/GCUtilities.cpp
        utils/GCTools.cpp
        utils/GCProgressDisplay.cpp
        utils/GCReconstructionUtils.cpp
        utils/Array.cpp
        utils/GCFileReader.cpp
        utils/GCGlobals.cpp)

# cuda
if (USE_CUDA)
    set(SOURCES "${SOURCES}"
            datastruct/scanner/GCScannerDevice.cu
            datastruct/image/ImageDevice.cu
            datastruct/image/ImageSpaceKernels.cu
            datastruct/projection/GCLORsDevice.cu
            datastruct/projection/GCProjectionDataDevice.cu
            datastruct/projection/GCProjectionSpaceKernels.cu
            operators/GCOperatorDevice.cu
            operators/GCOperatorProjectorDD_kernels.cu
            operators/GCOperatorProjectorDD_gpu.cu
            recon/GCOSEM_gpu.cu
            utils/GCPageLockedBuffer.cu
            utils/GCGPUStream.cu
            utils/GCGPUUtils.cu
    )
endif (USE_CUDA)

define_yrtpet_lib("${SOURCES}")

if (USE_CUDA)
    set_property(TARGET ${YRTPET_LIB_NAME} PROPERTY CUDA_ARCHITECTURES ${YRTPET_CUDA_ARCHITECTURES})
endif ()

# pybind111
set(SOURCES_PYBIND11
        pybind11.cpp)

if (BUILD_PYBIND11)
    define_target_pybind11(py${YRTPET_LIB_NAME} "${SOURCES_PYBIND11}" ${CMAKE_SOURCE_DIR})
endif (BUILD_PYBIND11)
