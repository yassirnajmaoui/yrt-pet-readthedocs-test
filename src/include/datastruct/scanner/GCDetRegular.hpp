/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "datastruct/scanner/GCDetectorSetup.hpp"
#include "datastruct/scanner/GCScanner.hpp"
#include "utils/Array.hpp"

#include <memory>

class GCDetRegular : public GCDetectorSetup
{
public:
	GCDetRegular(GCScanner* pp_scanner);
	void generateLUT();

	size_t getNumDets() const override;
	float getXpos(det_id_t detID) const override;
	float getYpos(det_id_t detID) const override;
	float getZpos(det_id_t detID) const override;
	float getXorient(det_id_t detID) const override;
	float getYorient(det_id_t detID) const override;
	float getZorient(det_id_t detID) const override;
	void writeToFile(const std::string& detCoord_fname) const override;
	// In case of a small modification after the generation,
	// We add the setters here
	virtual void setXpos(det_id_t detID, float f);
	virtual void setYpos(det_id_t detID, float f);
	virtual void setZpos(det_id_t detID, float f);
	virtual void setXorient(det_id_t detID, float f);
	virtual void setYorient(det_id_t detID, float f);
	virtual void setZorient(det_id_t detID, float f);

	Array1D<float>* getXposArrayRef() const { return (mp_Xpos.get()); }
	Array1D<float>* getYposArrayRef() const { return (mp_Ypos.get()); }
	Array1D<float>* getZposArrayRef() const { return (mp_Zpos.get()); }
	Array1D<float>* getXorientArrayRef() const { return (mp_Xorient.get()); }
	Array1D<float>* getYorientArrayRef() const { return (mp_Yorient.get()); }
	Array1D<float>* getZorientArrayRef() const { return (mp_Zorient.get()); }

	GCScanner* getScanner() { return mp_scanner; }
	virtual ~GCDetRegular() {}

protected:
	void allocate();

protected:
	std::unique_ptr<Array1D<float>> mp_Xpos;
	std::unique_ptr<Array1D<float>> mp_Ypos;
	std::unique_ptr<Array1D<float>> mp_Zpos;
	std::unique_ptr<Array1D<float>> mp_Xorient;
	std::unique_ptr<Array1D<float>> mp_Yorient;
	std::unique_ptr<Array1D<float>> mp_Zorient;
	GCScanner* mp_scanner;
};
