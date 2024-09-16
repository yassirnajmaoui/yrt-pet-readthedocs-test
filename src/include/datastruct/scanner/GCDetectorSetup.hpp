/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/GCVector.hpp"
#include "utils/GCTypes.hpp"

#include <string>

class GCDetectorSetup
{
public:
	virtual ~GCDetectorSetup() = default;
	virtual size_t getNumDets() const = 0;
	virtual float getXpos(det_id_t id) const = 0;
	virtual float getYpos(det_id_t id) const = 0;
	virtual float getZpos(det_id_t id) const = 0;
	virtual float getXorient(det_id_t id) const = 0;
	virtual float getYorient(det_id_t id) const = 0;
	virtual float getZorient(det_id_t id) const = 0;
	virtual void writeToFile(const std::string& detCoord_fname) const = 0;
	virtual GCVector getPos(det_id_t id) const;
	virtual GCVector getOrient(det_id_t id) const;
};
