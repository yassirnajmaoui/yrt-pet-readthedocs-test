/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "recon/GCVariable.hpp"

class GCOperator
{
public:
	virtual ~GCOperator() = default;
	virtual void applyA(const GCVariable* in, GCVariable* out) = 0;
	virtual void applyAH(const GCVariable* in, GCVariable* out) = 0;
};
