/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#include "utils/GCPageLockedBuffer.cuh"

#if BUILD_CUDA

#include <cuda.h>
#include <iostream>

template <typename T>
GCPageLockedBuffer<T>::GCPageLockedBuffer()
    : mph_dataPointer(nullptr),
      m_size(0ull),
      m_isPageLocked(false),
      m_currentFlags(0u)
{
}

template <typename T>
GCPageLockedBuffer<T>::GCPageLockedBuffer(const size_t size,
                                          const unsigned int flags)
    : GCPageLockedBuffer()
{
	allocate(size, flags);
}

template <typename T>
void GCPageLockedBuffer<T>::allocate(const size_t size,
                                     const unsigned int flags)
{
	cudaHostAlloc(reinterpret_cast<void**>(&mph_dataPointer), size * sizeof(T),
	              flags);
	const cudaError_t cudaError = cudaGetLastError();
	if (cudaError != 0)
	{
		std::cerr << "CUDA Error while allocating: "
		          << cudaGetErrorString(cudaError) << std::endl;
		mph_dataPointer = new T[size];
		m_isPageLocked = false;
	}
	else
	{
		m_isPageLocked = true;
		m_currentFlags = flags;
	}
	m_size = size;
}

template <typename T>
void GCPageLockedBuffer<T>::reAllocateIfNeeded(const size_t newSize,
                                               const unsigned int flags)
{
	if (newSize > m_size || m_currentFlags != flags)
	{
		allocate(newSize, flags);
	}
}
template <typename T>
void GCPageLockedBuffer<T>::deallocate()
{
	if (m_size > 0ull)
	{
		if (m_isPageLocked)
		{
			cudaFreeHost(mph_dataPointer);
			const cudaError_t cudaError = cudaGetLastError();
			if (cudaError != 0)
			{
				std::cerr << "CUDA Error while freeing: "
				          << cudaGetErrorString(cudaError) << std::endl;
			}
			else
			{
				m_size = 0ull;
			}
		}
		else
		{
			delete[] mph_dataPointer;
			m_size = 0ull;
		}
	}
}

template <typename T>
GCPageLockedBuffer<T>::~GCPageLockedBuffer()
{
	deallocate();
}

template <typename T>
T* GCPageLockedBuffer<T>::getPointer()
{
	return mph_dataPointer;
}

template <typename T>
const T* GCPageLockedBuffer<T>::getPointer() const
{
	return mph_dataPointer;
}

template <typename T>
size_t GCPageLockedBuffer<T>::getSize() const
{
	return m_size;
}

template class GCPageLockedBuffer<float>;
template class GCPageLockedBuffer<float3>;
template class GCPageLockedBuffer<float4>;
template class GCPageLockedBuffer<double>;
template class GCPageLockedBuffer<long>;
template class GCPageLockedBuffer<int>;
template class GCPageLockedBuffer<uint2>;
template class GCPageLockedBuffer<short>;
template class GCPageLockedBuffer<char>;

#endif
