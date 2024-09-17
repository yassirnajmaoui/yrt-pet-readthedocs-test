/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/GCStraightLineParam.hpp"

class Scanner;

class GCMultiRayGenerator
{
public:
	GCMultiRayGenerator(double thickness_z_i = 0.0,
	                    double thickness_trans_i = 0.0,
	                    bool isParallel_i = false);
	GCStraightLineParam getRandomLine(unsigned int& seed) const;
	void setupGenerator(const GCStraightLineParam& lor, const GCVector& n1,
	                    const GCVector& n2, const Scanner& scanner);

protected:
	double thickness_z, thickness_trans;
	bool isSingleRay;
	bool isParallel;

private:
	GCVector vect_parrallel_to_z;
	GCVector vect_parrallel_to_trans1;
	GCVector vect_parrallel_to_trans2;
	const GCStraightLineParam* currentLor;
};
