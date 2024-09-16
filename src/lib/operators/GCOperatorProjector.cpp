/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/GCOperatorProjector.hpp"

#include "datastruct/image/GCImage.hpp"
#include "datastruct/projection/GCBinIterator.hpp"
#include "datastruct/projection/GCHistogram3D.hpp"
#include "geometry/GCConstants.hpp"
#include "utils/GCAssert.hpp"
#include "utils/GCGlobals.hpp"
#include "utils/GCReconstructionUtils.hpp"
#include "utils/GCTools.hpp"

#include "omp.h"


#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

#include <utility>
namespace py = pybind11;

void py_setup_gcoperator(py::module& m)
{
	auto c = py::class_<GCOperator>(m, "GCOperator");
}

void py_setup_gcoperatorprojectorparams(py::module& m)
{
	auto c =
	    py::class_<GCOperatorProjectorParams>(m, "GCOperatorProjectorParams");
	c.def(py::init<GCBinIterator*, GCScanner*, float, int, const std::string&,
	               int>(),
	      py::arg("binIter"), py::arg("scanner"), py::arg("tofWidth_ps") = 0.f,
	      py::arg("tofNumStd") = 0, py::arg("psfProjFilename") = "",
	      py::arg("num_rays") = 1);
	c.def_readwrite("tofWidth_ps", &GCOperatorProjectorParams::tofWidth_ps);
	c.def_readwrite("tofNumStd", &GCOperatorProjectorParams::tofNumStd);
	c.def_readwrite("psfProjFilename",
	                &GCOperatorProjectorParams::psfProjFilename);
	c.def_readwrite("num_rays", &GCOperatorProjectorParams::numRays);
}

void py_setup_gcoperatorprojectorbase(py::module& m)
{
	auto c = py::class_<GCOperatorProjectorBase, GCOperator>(
	    m, "GCOperatorProjectorBase");
	c.def("setAddHisto", &GCOperatorProjectorBase::setAddHisto);
	c.def("setAttenuationImage", &GCOperatorProjectorBase::setAttenuationImage);
	c.def("setAttImage", &GCOperatorProjectorBase::setAttenuationImage);
	c.def("setAttImageForBackprojection",
	      &GCOperatorProjectorBase::setAttImageForBackprojection);
	c.def("getBinIter", &GCOperatorProjectorBase::getBinIter);
	c.def("getScanner", &GCOperatorProjectorBase::getScanner);
	c.def("getAttImage", &GCOperatorProjectorBase::getAttImage);
}

void py_setup_gcoperatorprojector(py::module& m)
{
	auto c = py::class_<GCOperatorProjector, GCOperatorProjectorBase>(
	    m, "GCOperatorProjector");
	c.def("setupTOFHelper", &GCOperatorProjector::setupTOFHelper);
	c.def("getTOFHelper", &GCOperatorProjector::getTOFHelper);
	c.def("getProjectionPsfManager",
	      &GCOperatorProjector::getProjectionPsfManager);
	c.def(
	    "applyA",
	    [](GCOperatorProjector& self, const GCImage* img, IProjectionData* proj)
	    { self.applyA(img, proj); }, py::arg("img"), py::arg("proj"));
	c.def(
	    "applyAH",
	    [](GCOperatorProjector& self, const IProjectionData* proj, GCImage* img)
	    { self.applyAH(proj, img); }, py::arg("proj"), py::arg("img"));

	py::enum_<GCOperatorProjector::ProjectorType>(c, "ProjectorType")
	    .value("SIDDON", GCOperatorProjector::ProjectorType::SIDDON)
	    .value("DD", GCOperatorProjector::ProjectorType::DD)
	    .value("DD_GPU", GCOperatorProjector::ProjectorType::DD_GPU)
	    .export_values();
}

#endif

GCOperatorProjectorParams::GCOperatorProjectorParams(
    const GCBinIterator* p_binIter, const GCScanner* p_scanner,
    float p_tofWidth_ps, int p_tofNumStd, std::string p_psfProjFilename,
    int p_num_rays)
    : binIter(p_binIter),
      scanner(p_scanner),
      tofWidth_ps(p_tofWidth_ps),
      tofNumStd(p_tofNumStd),
      psfProjFilename(std::move(p_psfProjFilename)),
      numRays(p_num_rays)
{
}

GCOperatorProjectorBase::GCOperatorProjectorBase(
    const GCOperatorProjectorParams& p_projParams)
    : binIter(p_projParams.binIter),
      scanner(p_projParams.scanner),
      attImage(nullptr),
      attImageForBackprojection(nullptr),
      addHisto(nullptr)
{
}

void GCOperatorProjectorBase::setAddHisto(const IHistogram* p_addHisto)
{
	ASSERT_MSG(p_addHisto != nullptr,
	           "The additive histogram given in "
	           "GCOperatorProjector::setAddHisto is a null pointer");
	addHisto = p_addHisto;
}

void GCOperatorProjectorBase::setBinIter(const GCBinIterator* p_binIter)
{
	binIter = p_binIter;
}

void GCOperatorProjectorBase::setAttenuationImage(const GCImage* p_attImage)
{
	setAttImage(p_attImage);
}

void GCOperatorProjectorBase::setAttImageForBackprojection(
    const GCImage* p_attImage)
{
	attImageForBackprojection = p_attImage;
}

void GCOperatorProjectorBase::setAttImage(const GCImage* p_attImage)
{
	ASSERT_MSG(p_attImage != nullptr,
	           "The attenuation image given in "
	           "GCOperatorProjector::setAttenuationImage is a null pointer");
	attImage = p_attImage;
}

const GCBinIterator* GCOperatorProjectorBase::getBinIter() const
{
	return binIter;
}

const GCScanner* GCOperatorProjectorBase::getScanner() const
{
	return scanner;
}

const GCImage* GCOperatorProjectorBase::getAttImage() const
{
	return attImage;
}

const GCImage* GCOperatorProjectorBase::getAttImageForBackprojection() const
{
	return attImageForBackprojection;
}

const IHistogram* GCOperatorProjectorBase::getAddHisto() const
{
	return addHisto;
}

GCOperatorProjector::GCOperatorProjector(
    const GCOperatorProjectorParams& p_projParams)
    : GCOperatorProjectorBase(p_projParams),
      mp_tofHelper(nullptr),
      mp_projPsfManager(nullptr)
{
	if (p_projParams.tofWidth_ps > 0.f)
	{
		setupTOFHelper(p_projParams.tofWidth_ps, p_projParams.tofNumStd);
	}
	if (!p_projParams.psfProjFilename.empty())
	{
		setupProjPsfManager(p_projParams.psfProjFilename);
	}
	ASSERT_MSG_WARNING(
	    mp_projPsfManager == nullptr,
	    "Siddon does not support Projection space PSF. It will be ignored.");
}

void GCOperatorProjector::applyA(const GCVariable* in, GCVariable* out)
{
	auto* dat = dynamic_cast<IProjectionData*>(out);
	auto* img = dynamic_cast<const GCImage*>(in);

	ASSERT_MSG(dat != nullptr, "Input variable has to be Projection data");
	ASSERT_MSG(img != nullptr, "Output variable has to be an image");
#pragma omp parallel for
	for (bin_t binIdx = 0; binIdx < binIter->size(); binIdx++)
	{
		const bin_t bin = binIter->get(binIdx);

		double imProj = 0.f;
		imProj += forwardProjection(img, dat, bin);

		if (attImage != nullptr)
		{
			// Multiplicative attenuation correction (for motion)
			const double attProj = forwardProjection(attImage, dat, bin);
			const double attProj_coeff =
			    Util::getAttenuationCoefficientFactor(attProj);
			imProj *= attProj_coeff;
		}

		if (addHisto != nullptr)
		{
			// Additive correction(s)
			const histo_bin_t histoBin = dat->getHistogramBin(bin);
			imProj += addHisto->getProjectionValueFromHistogramBin(histoBin);
		}
		dat->setProjectionValue(bin, static_cast<float>(imProj));
	}
}

void GCOperatorProjector::applyAH(const GCVariable* in, GCVariable* out)
{
	auto* dat = dynamic_cast<const IProjectionData*>(in);
	auto* img = dynamic_cast<GCImage*>(out);
	ASSERT(dat != nullptr && img != nullptr);

#pragma omp parallel for
	for (bin_t binIdx = 0; binIdx < binIter->size(); binIdx++)
	{
		const bin_t bin = binIter->get(binIdx);

		double projValue = dat->getProjectionValue(bin);
		if (std::abs(projValue) < SMALL)
		{
			continue;
		}

		if (attImageForBackprojection != nullptr)
		{
			// Multiplicative attenuation correction
			const double attProj =
			    forwardProjection(attImageForBackprojection, dat, bin);
			const double attProj_coeff =
			    Util::getAttenuationCoefficientFactor(attProj);
			projValue *= attProj_coeff;
		}

		backProjection(img, dat, bin, projValue);
	}
}

void GCOperatorProjector::setupTOFHelper(float tofWidth_ps, int tofNumStd)
{
	mp_tofHelper =
	    std::make_unique<GCTimeOfFlightHelper>(tofWidth_ps, tofNumStd);
	ASSERT_MSG(mp_tofHelper != nullptr,
	           "Error occured during the setup of GCTimeOfFlightHelper");
}

void GCOperatorProjector::setupProjPsfManager(const std::string& psfFilename)
{
	mp_projPsfManager = std::make_unique<GCProjectionPsfManager>(psfFilename);
	ASSERT_MSG(mp_projPsfManager != nullptr,
	           "Error occured during the setup of GCProjectionPsfManager");
}

const GCTimeOfFlightHelper* GCOperatorProjector::getTOFHelper() const
{
	return mp_tofHelper.get();
}

const GCProjectionPsfManager*
    GCOperatorProjector::getProjectionPsfManager() const
{
	return mp_projPsfManager.get();
}

void GCOperatorProjector::get_alpha(double r0, double r1, double p1, double p2,
                                    double inv_p12, double& amin, double& amax)
{
	amin = 0.0;
	amax = 1.0;
	if (p1 != p2)
	{
		const double a0 = (r0 - p1) * inv_p12;
		const double a1 = (r1 - p1) * inv_p12;
		if (a0 < a1)
		{
			amin = a0;
			amax = a1;
		}
		else
		{
			amin = a1;
			amax = a0;
		}
	}
	else if (p1 < r0 || p1 > r1)
	{
		amax = 0.0;
		amin = 1.0;
	}
}
