/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */
#include "utils/GCFileReader.hpp"

#include <algorithm>

namespace Util
{
	GCFileReader::GCFileReader(std::istream& pr_istream, bool p_useCache,
	                           size_t p_cacheSize)
	    : m_cacheStart(-1),
	      mr_istream(pr_istream),
	      m_foundEof(false),
	      m_useCache(p_useCache)
	{
		if (isUsingCache())
		{
			m_cache.allocate(p_cacheSize);
			m_cacheSize = static_cast<std::streamsize>(m_cache.GetSize(0));
		}
	}

	std::streamsize GCFileReader::read(std::streamoff startPos,
	                                   char* receivingBuffer,
	                                   std::streamsize bytesToRead)
	{
		if (isUsingCache())
		{
			// Check if hit or miss
			bool cacheHit = m_cacheStart >= 0;     // Before first reading
			cacheHit &= startPos >= m_cacheStart;  // First byte
			cacheHit &= (startPos + bytesToRead) <=
			            (m_cacheStart + m_cacheSize);  // Last byte
			if (bytesToRead > m_cacheSize)
			{
				throw std::range_error("The requested number of bytes to read "
				                       "exceeds the cache size");
			}

			if (!cacheHit)
			{
				if (m_foundEof)
				{
					std::cerr << "Warning: the file might be incomplete"
					          << std::endl;
				}
				else
				{
					m_foundEof = readStreamToCache(startPos);
				}
			}

			std::streamoff offset = (startPos - m_cacheStart);
			char* initPos = m_cache.GetRawPointer() + offset;
			char* lastPos = std::copy_n(
			    initPos, std::min(m_cacheSize - offset, bytesToRead),
			    receivingBuffer);
			// Return the number of bytes copied
			return lastPos - receivingBuffer;
		}
		// simple read
		const std::streamsize bytesRead =
		    mr_istream.read(receivingBuffer, bytesToRead).gcount();
		m_foundEof = mr_istream.eof();
		return bytesRead;
	}

	bool GCFileReader::readStreamToCache(std::streamoff startPos)
	{
		m_cacheStart = startPos;
		mr_istream.seekg(startPos, std::ios::beg);
		if (mr_istream.fail())
		{
			throw std::range_error(
			    "Error reading file: Either the seeked position"
			    "is outside the file or the file is corrupt");
		}
		mr_istream.read(m_cache.GetRawPointer(), m_cacheSize);
		if (mr_istream.eof())
		{
			m_cacheSize = mr_istream.gcount();
			return true;
		}
		return false;
	}

	bool GCFileReader::foundEof() const
	{
		return m_foundEof;
	}

	std::streamsize GCFileReader::cacheSize() const
	{
		return m_cacheSize;
	}

	std::streamoff GCFileReader::cacheStart() const
	{
		return m_cacheStart;
	}

	bool GCFileReader::isUsingCache() const
	{
		return m_useCache;
	}

	GCFileReaderContiguous::GCFileReaderContiguous(std::istream& pr_istream,
	                                               bool p_useCache,
	                                               size_t p_cacheSize)
	    : GCFileReader(pr_istream, p_useCache, p_cacheSize), m_readPos(0ull)
	{
	}

	bool GCFileReaderContiguous::finishedReading() const
	{
		return foundEof() &&
		       (!isUsingCache() || (m_readPos >= cacheStart() + cacheSize()));
	}

	std::streamsize GCFileReaderContiguous::read(std::streamoff startPos,
	                                             char* receivingBuffer,
	                                             std::streamsize bytesToRead)
	{
		(void)startPos;
		(void)receivingBuffer;
		(void)bytesToRead;
		throw std::runtime_error(
		    "Arbitrary access unsupported on contiguous reading");
	}

	std::streamsize GCFileReaderContiguous::read(char* receivingBuffer,
	                                             std::streamsize bytesToRead)
	{
		const std::streamsize bytesRead =
		    GCFileReader::read(m_readPos, receivingBuffer, bytesToRead);
		m_readPos += bytesRead;
		return bytesRead;
	}

}  // namespace Util
