/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "geometry/GCStraightLineParam.hpp"
#include "operators/GCTimeOfFlight.hpp"

#include "catch.hpp"
#include <cmath>
#include <iostream>

TEST_CASE("TOF", "[tof]")
{
	int random_seed = time(0);
	srand(random_seed);
	std::string rseed_str = "random_seed=" + std::to_string(random_seed);
	INFO(rseed_str);

	SECTION("width_0")
	{
		auto tofHelper = GCTimeOfFlightHelper(0.f, 3);
		GCStraightLineParam lor(
		    GCVector(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX,
		             rand() / (float)RAND_MAX),
		    GCVector(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX,
		             rand() / (float)RAND_MAX));
		double d_norm = (lor.point2 - lor.point1).getNorm();
		float tof_value_ps = (rand() / (float)RAND_MAX) * 0.5f - 0.5f;
		double amin, amax;
		tofHelper.getAlphaRange(amin, amax, d_norm, tof_value_ps);
		REQUIRE(amin == 0.);
		REQUIRE(amax == 1.);
	}

	SECTION("simple_geom")
	{
		float tof_width_ps = 95.f;
		auto tofHelper = GCTimeOfFlightHelper(tof_width_ps, 3);
		GCStraightLineParam lor(GCVector(-155.0f, 118.75f, 367.5f),
		                        GCVector(155.0f, 163.75f, 373.5f));
		double d_norm = (lor.point2 - lor.point1).getNorm();
		float tof_value_ps = -408.f;
		float pix_pos_lo = 0.24 * d_norm;
		float pix_pos_hi = 0.25 * d_norm;
		float tof_weight =
		    tofHelper.getWeight(d_norm, tof_value_ps, pix_pos_lo, pix_pos_hi);
		REQUIRE(fabs(tof_weight - 0.000543244) < 1e-4);
	}
}
