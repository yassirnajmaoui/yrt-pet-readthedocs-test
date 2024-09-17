project(lib)

# List source files explicitly
set(SOURCES
        datastruct/projection/Histogram.cpp
        datastruct/projection/Histogram3D.cpp
        datastruct/projection/UniformHistogram.cpp
        datastruct/projection/ListModeLUT.cpp
        datastruct/projection/ListModeLUTDOI.cpp
        datastruct/projection/LORMotion.cpp
        datastruct/projection/BinIterator.cpp
        datastruct/projection/ProjectionList.cpp
        datastruct/projection/SparseHistogram.cpp
        datastruct/projection/ProjectionData.cpp
        datastruct/projection/ListMode.cpp
        datastruct/scanner/DetCoord.cpp
        datastruct/scanner/DetRegular.cpp
        datastruct/scanner/DetectorSetup.cpp
        datastruct/scanner/Scanner.cpp
        datastruct/image/ImageBase.cpp
        datastruct/image/Image.cpp
        datastruct/IO.cpp
        geometry/StraightLineParam.cpp
        geometry/Matrix.cpp
        geometry/Vector3D.cpp
        geometry/MultiRayGenerator.cpp
        geometry/TubeOfResponse.cpp
        geometry/Cylinder.cpp
        geometry/Plane.cpp
        kernel/Kernel.cpp
        kinetic/SRTM.cpp
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
            datastruct/scanner/ScannerDevice.cu
            datastruct/image/ImageDevice.cu
            datastruct/image/ImageSpaceKernels.cu
            datastruct/projection/LORsDevice.cu
            datastruct/projection/ProjectionDataDevice.cu
            datastruct/projection/ProjectionSpaceKernels.cu
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
