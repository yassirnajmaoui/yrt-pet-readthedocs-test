/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/GCGPUStream.cuh"

GCGPUStream::GCGPUStream(unsigned int flags)
{
	cudaStreamCreateWithFlags(&m_stream, flags);
}

const cudaStream_t& GCGPUStream::getStream() const
{
	return m_stream;
}

GCGPUStream::~GCGPUStream()
{
	cudaStreamDestroy(m_stream);
}
