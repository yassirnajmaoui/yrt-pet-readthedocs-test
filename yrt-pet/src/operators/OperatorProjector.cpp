/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "operators/OperatorProjector.hpp"

#include "datastruct/image/Image.hpp"
#include "datastruct/projection/BinIterator.hpp"
#include "datastruct/projection/Histogram3D.hpp"
#include "geometry/Constants.hpp"
#include "utils/Assert.hpp"
#include "utils/Globals.hpp"
#include "utils/ReconstructionUtils.hpp"
#include "utils/Tools.hpp"

#include "omp.h"


#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

#include <utility>
namespace py = pybind11;


void py_setup_operatorprojector(py::module& m)
{
	auto c = py::class_<OperatorProjector, OperatorProjectorBase>(
	    m, "OperatorProjector");
	c.def("setupTOFHelper", &OperatorProjector::setupTOFHelper);
	c.def("getTOFHelper", &OperatorProjector::getTOFHelper);
	c.def("getProjectionPsfManager",
	      &OperatorProjector::getProjectionPsfManager);
	c.def(
	    "applyA",
	    [](OperatorProjector& self, const Image* img, ProjectionData* proj)
	    { self.applyA(img, proj); }, py::arg("img"), py::arg("proj"));
	c.def(
	    "applyAH",
	    [](OperatorProjector& self, const ProjectionData* proj, Image* img)
	    { self.applyAH(proj, img); }, py::arg("proj"), py::arg("img"));

	py::enum_<OperatorProjector::ProjectorType>(c, "ProjectorType")
	    .value("SIDDON", OperatorProjector::ProjectorType::SIDDON)
	    .value("DD", OperatorProjector::ProjectorType::DD)
	    .export_values();
}

#endif

OperatorProjector::OperatorProjector(const Scanner& pr_scanner,
                                     float tofWidth_ps, int tofNumStd,
                                     const std::string& projPsf_fname)
    : OperatorProjectorBase{pr_scanner},
      mp_tofHelper{nullptr},
      mp_projPsfManager{nullptr}
{
	if (tofWidth_ps > 0.0f)
	{
		setupTOFHelper(tofWidth_ps, tofNumStd);
	}
	if (!projPsf_fname.empty())
	{
		setupProjPsfManager(projPsf_fname);
	}
}

OperatorProjector::OperatorProjector(
    const OperatorProjectorParams& p_projParams)
    : OperatorProjectorBase{p_projParams},
      mp_tofHelper{nullptr},
      mp_projPsfManager{nullptr}
{
	if (p_projParams.tofWidth_ps > 0.f)
	{
		setupTOFHelper(p_projParams.tofWidth_ps, p_projParams.tofNumStd);
	}
	if (!p_projParams.projPsf_fname.empty())
	{
		setupProjPsfManager(p_projParams.projPsf_fname);
	}
}

void OperatorProjector::applyA(const Variable* in, Variable* out)
{
	auto* dat = dynamic_cast<ProjectionData*>(out);
	auto* img = dynamic_cast<const Image*>(in);

	ASSERT_MSG(dat != nullptr, "Output variable has to be Projection data");
	ASSERT_MSG(img != nullptr, "Input variable has to be an Image");
	ASSERT_MSG(binIter != nullptr, "BinIterator undefined");

#pragma omp parallel for default(none) firstprivate(binIter, img, dat)
	for (bin_t binIdx = 0; binIdx < binIter->size(); binIdx++)
	{
		const bin_t bin = binIter->get(binIdx);

		ProjectionProperties projectionProperties =
		    dat->getProjectionProperties(bin);

		if (projectionProperties.lor.point1.x !=
		    std::numeric_limits<float>::infinity())
		{
			float imProj = forwardProjection(img, projectionProperties);

			dat->setProjectionValue(bin, static_cast<float>(imProj));
		}
	}
}

void OperatorProjector::applyAH(const Variable* in, Variable* out)
{
	auto* dat = dynamic_cast<const ProjectionData*>(in);
	auto* img = dynamic_cast<Image*>(out);

	ASSERT_MSG(dat != nullptr, "Input variable has to be Projection data");
	ASSERT_MSG(img != nullptr, "Output variable has to be an Image");
	ASSERT_MSG(binIter != nullptr, "BinIterator undefined");

#pragma omp parallel for default(none) firstprivate(binIter, img, dat)
	for (bin_t binIdx = 0; binIdx < binIter->size(); binIdx++)
	{
		const bin_t bin = binIter->get(binIdx);

		ProjectionProperties projectionProperties =
		    dat->getProjectionProperties(bin);

		if (projectionProperties.lor.point1.x !=
		    std::numeric_limits<float>::infinity())
		{
			// TODO: What to do with randomsEstimate ?

			float projValue = dat->getProjectionValue(bin);
			if (std::abs(projValue) < SMALL)
			{
				continue;
			}

			backProjection(img, projectionProperties, projValue);
		}
	}
}

void OperatorProjector::addTOF(float tofWidth_ps, int tofNumStd)
{
	setupTOFHelper(tofWidth_ps, tofNumStd);
}

void OperatorProjector::setupTOFHelper(float tofWidth_ps, int tofNumStd)
{
	mp_tofHelper = std::make_unique<TimeOfFlightHelper>(tofWidth_ps, tofNumStd);
	ASSERT_MSG(mp_tofHelper != nullptr,
	           "Error occured during the setup of TimeOfFlightHelper");
}

void OperatorProjector::setupProjPsfManager(const std::string& projPsf_fname)
{
	mp_projPsfManager = std::make_unique<ProjectionPsfManager>(projPsf_fname);
	ASSERT_MSG(mp_projPsfManager != nullptr,
	           "Error occured during the setup of ProjectionPsfManager");
}

const TimeOfFlightHelper* OperatorProjector::getTOFHelper() const
{
	return mp_tofHelper.get();
}

const ProjectionPsfManager* OperatorProjector::getProjectionPsfManager() const
{
	return mp_projPsfManager.get();
}
