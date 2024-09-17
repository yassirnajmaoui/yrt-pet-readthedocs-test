/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/image/Image.hpp"
#include "operators/GCOperatorProjector.hpp"
#include "operators/GCOperatorPsf.hpp"

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>
#endif

class ImageWarperTemplate;

class GCOSEM
{
public:
	// ---------- Constants ----------
	static constexpr int DEFAULT_NUM_ITERATIONS = 10;
	static constexpr float DEFAULT_HARD_THRESHOLD = 1.0f;
	static constexpr float INITIAL_VALUE_MLEM = 0.1f;
	// ---------- Public methods ----------
	GCOSEM(const Scanner* p_scanner);
	virtual ~GCOSEM() = default;
	GCOSEM(const GCOSEM&) = delete;
	GCOSEM& operator=(const GCOSEM&) = delete;

	// Sensitivity image generation
	void generateSensitivityImages(const std::string& out_fname);
	void generateSensitivityImages(
	    std::vector<std::unique_ptr<Image>>& sensImages,
	    const std::string& out_fname);
	bool validateSensImagesAmount(int size) const;

	// In case the sensitivity images were already generated
	void registerSensitivityImages(
	    const std::vector<std::unique_ptr<Image>>& sensImages);
#if BUILD_PYBIND11
	void registerSensitivityImages(pybind11::list& imageList);
#endif

	// OSEM Reconstruction
	void reconstruct();
	void reconstructWithWarperMotion();

	// Prints a summary of the parameters
	void summary() const;

	// ---------- Getters and setters ----------
	ProjectionData* getSensDataInput() { return sensDataInput; }
	const ProjectionData* getSensDataInput() const { return sensDataInput; }
	void setSensDataInput(ProjectionData* p_sensDataInput);
	ProjectionData* getDataInput() { return dataInput; }
	const ProjectionData* getDataInput() const { return dataInput; }
	void setDataInput(ProjectionData* p_dataInput);
	void addTOF(float p_tofWidth_ps, int p_tofNumStd);
	void addProjPSF(const std::string& p_projSpacePsf_fname);
	void addImagePSF(GCOperatorPsf* p_imageSpacePsf);
	void setSaveSteps(int p_saveSteps, const std::string& p_saveStepsPath);
	void setListModeEnabled(bool enabled);
	void setProjector(const std::string& projectorName); // Helper
	bool isListModeEnabled() const;

	// ---------- Public members ----------
	int num_MLEM_iterations;
	int num_OSEM_subsets;
	float hardThreshold;
	int numRays;  // For Siddon only
	GCOperatorProjector::ProjectorType projectorType;
	ImageParams imageParams;
	const Scanner* scanner;
	const Image* maskImage;
	const Image* attenuationImage;
	const Image* attenuationImageForBackprojection;
	const Histogram* addHis;
	ImageWarperTemplate* warper; // For MLEM with Warper only
	// TODO: Maybe make it so a buffer is automatically created in Initialize
	Image* outImage;  // Buffer to for recon fill (Note: This is a host image)

protected:
	// ---------- Internal Getters ----------
	auto& getBinIterators() { return m_binIterators; }
	const auto& getBinIterators() const { return m_binIterators; }

	auto& getProjector() { return mp_projector; }
	const auto& getProjector() const { return mp_projector; }

	const ImageParams& getImageParams() const { return imageParams; }

	const Image* getSensitivityImage(int subsetId) const;
	Image* getSensitivityImage(int subsetId);

	// ---------- Protected members ----------
	bool flagImagePSF;
	GCOperatorPsf* imageSpacePsf;
	bool flagProjPSF;
	std::string projSpacePsf_fname;
	bool flagProjTOF;
	float tofWidth_ps;
	int tofNumStd;
	int saveSteps;
	std::string saveStepsPath;
	bool usingListModeInput;  // true => ListMode, false => Histogram
	std::unique_ptr<GCOperatorProjectorBase> mp_projector;

	// ---------- Virtual pure functions ----------

	// Sens Image generator driver
	virtual void SetupOperatorsForSensImgGen() = 0;
	virtual void allocateForSensImgGen() = 0;
	virtual std::unique_ptr<Image> GetLatestSensitivityImage(bool isLastSubset) = 0;
	virtual void EndSensImgGen() = 0;

	// Reconstruction driver
	virtual void SetupOperatorsForRecon() = 0;
	virtual void allocateForRecon() = 0;
	virtual void EndRecon() = 0;
	virtual void CompleteMLEMIteration() = 0;

	// Abstract Getters
	virtual ImageBase* GetSensImageBuffer() = 0;
	virtual ProjectionData* GetSensDataInputBuffer() = 0;
	virtual ImageBase* GetMLEMImageBuffer() = 0;
	virtual ImageBase* GetMLEMImageTmpBuffer() = 0;
	virtual ProjectionData* GetMLEMDataBuffer() = 0;
	virtual ProjectionData* GetMLEMDataTmpBuffer() = 0;
	virtual int GetNumBatches(int subsetId, bool forRecon) const;

	// Common methods
	virtual void LoadBatch(int p_batchId, bool p_forRecon) = 0;
	virtual void LoadSubset(int p_subsetId, bool p_forRecon) = 0;

private:
	void LoadSubsetInternal(int p_subsetId, bool p_forRecon);
	void InitializeForSensImgGen();
	void GenerateSensitivityImageForSubset(int subsetId);
	void GenerateSensitivityImagesCore(
	    bool saveOnDisk, const std::string& out_fname, bool saveOnMemory,
	    std::vector<std::unique_ptr<Image>>& sensImages);
	void InitializeForRecon();

	std::vector<std::unique_ptr<BinIterator>> m_binIterators;

	ProjectionData* sensDataInput;
	ProjectionData* dataInput;

	std::vector<Image*> sensitivityImages;
};
