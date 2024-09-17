/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "geometry/StraightLineParam.hpp"

class Scanner;

class MultiRayGenerator
{
public:
	MultiRayGenerator(double thickness_z_i = 0.0,
	                    double thickness_trans_i = 0.0,
	                    bool isParallel_i = false);
	StraightLineParam getRandomLine(unsigned int& seed) const;
	void setupGenerator(const StraightLineParam& lor, const Vector3D& n1,
	                    const Vector3D& n2, const Scanner& scanner);

protected:
	double thickness_z, thickness_trans;
	bool isSingleRay;
	bool isParallel;

private:
	Vector3D vect_parrallel_to_z;
	Vector3D vect_parrallel_to_trans1;
	Vector3D vect_parrallel_to_trans2;
	const StraightLineParam* currentLor;
};
