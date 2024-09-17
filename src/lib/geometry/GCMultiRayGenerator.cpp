/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/GCMultiRayGenerator.hpp"

#include "datastruct/scanner/Scanner.hpp"
#include "geometry/GCConstants.hpp"

#include <cmath>
#include <utility>
#include <vector>

GCMultiRayGenerator::GCMultiRayGenerator(double thickness_z_i,
                                         double thickness_trans_i,
                                         bool isParallel_i)
    : thickness_z(thickness_z_i),
      thickness_trans(thickness_trans_i),
      isParallel(isParallel_i),
      currentLor(nullptr)
{
	isSingleRay = (thickness_trans <= 0 && thickness_z <= 0);
	vect_parrallel_to_z = GCVector(0, 0, 1);
}

void GCMultiRayGenerator::setupGenerator(const GCStraightLineParam& lor,
                                         const GCVector& n1, const GCVector& n2,
                                         const Scanner& scanner)
{
	currentLor = &lor;
	if (!isSingleRay)
	{
		vect_parrallel_to_trans1 =
		    n1.crossProduct(vect_parrallel_to_z).normalize() * scanner.crystalSize_trans;
		vect_parrallel_to_trans2 =
		    n2.crossProduct(vect_parrallel_to_z).normalize() * scanner.crystalSize_trans;
	}
}

GCStraightLineParam GCMultiRayGenerator::getRandomLine(unsigned int& seed) const
{
	if (isSingleRay)
	{
		return *currentLor;
	}
	const double rand_i_1 =
	    static_cast<double>(rand_r(&seed)) / static_cast<double>(RAND_MAX) -
	    0.5;
	const double rand_j_1 =
	    static_cast<double>(rand_r(&seed)) / static_cast<double>(RAND_MAX) -
	    0.5;

	const double rand_i_2 = (isParallel) ?
	                            rand_i_1 :
	                            (static_cast<double>(rand_r(&seed)) / RAND_MAX);
	const double rand_j_2 = (isParallel) ?
	                            -rand_j_1 :
	                            (static_cast<double>(rand_r(&seed)) / RAND_MAX);

	const GCVector pt1 =
	    currentLor->point1 + vect_parrallel_to_z * (rand_i_1 * thickness_z) +
	    vect_parrallel_to_trans1 * (rand_j_1 * thickness_trans);
	const GCVector pt2 =
	    currentLor->point2 + vect_parrallel_to_z * (rand_i_2 * thickness_z) +
	    vect_parrallel_to_trans2 * (rand_j_2 * thickness_trans);

	return GCStraightLineParam{pt1, pt2};
}
