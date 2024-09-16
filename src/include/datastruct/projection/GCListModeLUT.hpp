/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/PluginFramework.hpp"
#include "datastruct/projection/GCLORMotion.hpp"
#include "datastruct/projection/IListMode.hpp"
#include "geometry/GCStraightLineParam.hpp"
#include "utils/Array.hpp"

#if BUILD_PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#endif

class GCScanner;

class GCListModeLUT : public IListMode
{
public:
	// Methods
	~GCListModeLUT() override = default;

	timestamp_t getTimestamp(bin_t eventId) const override;
	det_id_t getDetector1(bin_t eventId) const override;
	det_id_t getDetector2(bin_t eventId) const override;
	GCStraightLineParam getNativeLORFromId(bin_t id) const;
	bool hasTOF() const override;
	float getTOFValue(bin_t id) const override;
	size_t count() const override;
	bool isUniform() const override;
	bool hasMotion() const override;
	frame_t getFrame(bin_t id) const override;
	size_t getNumFrames() const override;
	transform_t getTransformOfFrame(frame_t frame) const override;

	void setDetectorId1OfEvent(bin_t eventId, det_id_t d1);
	void setDetectorId2OfEvent(bin_t eventId, det_id_t d2);
	void setDetectorIdsOfEvent(bin_t eventId, det_id_t d1, det_id_t d2);
	const GCScanner* getScanner() const;

	Array1DBase<timestamp_t>* getTimestampArrayPtr() const;
	Array1DBase<det_id_t>* getDetector1ArrayPtr() const;
	Array1DBase<det_id_t>* getDetector2ArrayPtr() const;

	virtual void writeToFile(const std::string& listMode_fname) const;

	void addLORMotion(const std::string& lorMotion_fname);

protected:
	GCListModeLUT(const GCScanner* s, bool p_flagTOF = false);

	// Parameters
	// The detector Id of the events.
	const GCScanner* mp_scanner;
	// TODO: Replace getTimestamp by getFrame.
	//  Replace this array by an array of frames
	//  Repopulate this array with frame ids after lorMotion is added
	std::unique_ptr<Array1DBase<timestamp_t>> mp_timestamps;
	std::unique_ptr<Array1DBase<det_id_t>> mp_detectorId1;
	std::unique_ptr<Array1DBase<det_id_t>> mp_detectorId2;
	bool m_flagTOF;
	std::unique_ptr<Array1DBase<float>> mp_tof_ps;

	std::unique_ptr<GCLORMotion> mp_lorMotion;
	std::unique_ptr<Array1D<frame_t>> mp_frames;
};


class GCListModeLUTAlias : public GCListModeLUT
{
public:
	GCListModeLUTAlias(GCScanner* s, bool p_flagTOF = false);
	~GCListModeLUTAlias() override = default;
	void Bind(GCListModeLUT* listMode);
	void Bind(Array1DBase<timestamp_t>* p_timestamps,
	          Array1DBase<det_id_t>* p_detector_ids1,
	          Array1DBase<det_id_t>* p_detector_ids2,
	          Array1DBase<float>* p_tof_ps = nullptr);
#if BUILD_PYBIND11
	void Bind(
	    pybind11::array_t<timestamp_t, pybind11::array::c_style>& p_timestamps,
	    pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detector_ids1,
	    pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detector_ids2);
	void Bind(
	    pybind11::array_t<timestamp_t, pybind11::array::c_style>& p_timestamps,
	    pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detector_ids1,
	    pybind11::array_t<det_id_t, pybind11::array::c_style>& p_detector_ids2,
	    pybind11::array_t<float, pybind11::array::c_style>& p_tof_ps);
#endif
};


class GCListModeLUTOwned : public GCListModeLUT
{
public:
	GCListModeLUTOwned(const GCScanner* s, bool p_flagTOF = false);
	GCListModeLUTOwned(const GCScanner* s, const std::string& listMode_fname,
	                   bool p_flagTOF = false);
	~GCListModeLUTOwned() override = default;

	void readFromFile(const std::string& listMode_fname);
	void allocate(size_t numEvents);

	// For registering the plugin
	static std::unique_ptr<IProjectionData>
	    create(const GCScanner& scanner, const std::string& filename,
	           const Plugin::OptionsResult& pluginOptions);
	static Plugin::OptionsListPerPlugin getOptions();
};
