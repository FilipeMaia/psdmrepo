#ifndef PDS_CODEC_STREAMCOMPRESSOR_H
#define PDS_CODEC_STREAMCOMPRESSOR_H

//--------------------------------------------------------------------------
// File and Version Information:
//   $Id: $
//
// Description:
//   Class StreamCompressor. A utility wrapper allowing multithreaded
//   compression of a data stream using API conforming algorithms and
//   multi-thread wrappers. The wrapper will optionally (if requested)
//   split the input stream into segments of the specified size and
//   compress each segment independently, possibly in a separate thread.
//
// Author:
//   Igor A. Gaponenko, SLAC National Accelerator Laboratory
//--------------------------------------------------------------------------

// DEBUG NOTE: Please, uncomment the following line to see the diagnostic
//             print outs of the alforithm.
//
//#define PDS_CODEC_STREAMCOMPRESSOR_DEBUG

//-----------------
// C/C++ Headers --
//-----------------

#include <stdio.h>
#include <assert.h>
#include <string.h>

#ifdef PDS_CODEC_STREAMCOMPRESSOR_DEBUG
#include <iostream>
#include <iomanip>
#endif

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

namespace Pds {

    namespace Codec {

        template< class COMPRESSOR, template< class COMPRESSOR > class MT >
        class StreamCompressor {

        public:

            enum {

                Success = 0,

                // Status values returned due to incorrect input to operations
                //
                ErrNotSupportedSize,  // to report incompatibility between buffer or segment sizes and a compressor type.
                                      // Note that some compressors will work only on certain types of data (bytes,
                                      // 16-bit integers, etc.).
                ErrNoData,            // no data to process
                ErrBadFormat,         // unsupported or bad format of the input buffer

                // Status values returned by a specified algorithm or a MT wrapper
                //
                ErrInCompressor,      // a problem reported by the compressor
                ErrInMT               // a problem reported by the MT wrapper
            };

            explicit StreamCompressor(size_t segmentSize=0, size_t numThreads=1);

            virtual ~StreamCompressor();

            int compress(const void*   inData, size_t   inDataSize,
                               void*& outData, size_t& outDataSize, int*& stat, size_t& outNumSegments);

            int decompress(const void* outData, const size_t outDataSize,
                                 void*& inData,       size_t& inDataSize, int*& stat, size_t& inNumSegments);

            static const char* err2str(int code);

        private:

            size_t m_segmentSize;

            MT<COMPRESSOR >* m_compressor;

            // Buffers for storing results of the operations. Pointers to the buffers
            // will be rturned by the corresponding method. Buffers may be dynamically
            // reallocated between requests to accomodate larger results.
            //
            void*  m_inData;       // the buffer for storing result of the most recent decompression
            size_t m_inDataSize;   // the buffers size (bytes) from the largest decompression made in the past

            void*  m_outData;      // the buffer for storing result of the most recent compression
            size_t m_outDataSize;  // the buffers size (bytes) from the largest compression made in the past

            // Buffers for intermediate data at a segment level.  Buffers may be dynamically
            // reallocated between requests to accomodate larger results.
            //
            void**                            m_segmentInData;        // pointers to uncompressed segments
            typename COMPRESSOR::ImageParams* m_segmentInDataParams;  // parameters of uncompressed segments

            void**      m_segmentOutData;      // pointers to compressed segments
            size_t*     m_segmentOutDataSize;  // sizes of compressed segments

            int*        m_segmentStat;  // the array has m_numSegments elements

            size_t      m_numSegments;  // the largest number of segments from the past requests
        };
    }
}

template< class COMPRESSOR, template< class COMPRESSOR > class MT >
Pds::Codec::StreamCompressor<COMPRESSOR,MT>::StreamCompressor(size_t segmentSize, size_t numThreads) :

    m_segmentSize(segmentSize),
    m_compressor (new MT<COMPRESSOR>( numThreads )),

    m_inData    (0),
    m_inDataSize(0),

    m_outData    (0),
    m_outDataSize(0),

    m_segmentInData      (0),
    m_segmentInDataParams(0),

    m_segmentOutData    (0),
    m_segmentOutDataSize(0),

    m_segmentStat(0),

    m_numSegments(0)
{}

template< class COMPRESSOR, template< class COMPRESSOR > class MT >
Pds::Codec::StreamCompressor<COMPRESSOR,MT>::~StreamCompressor()
{
    delete m_compressor; m_compressor = 0;

    delete [] (uint8_t*)m_inData;  m_inData  = 0;
    delete [] (uint8_t*)m_outData; m_outData = 0;

    delete [] m_segmentInData;       m_segmentInData       = 0;
    delete [] m_segmentInDataParams; m_segmentInDataParams = 0;

    delete [] m_segmentOutData;     m_segmentOutData     = 0;
    delete [] m_segmentOutDataSize; m_segmentOutDataSize = 0;

    delete [] m_segmentStat; m_segmentStat        = 0;
}

template< class COMPRESSOR, template< class COMPRESSOR > class MT >
int
Pds::Codec::StreamCompressor<COMPRESSOR,MT>::compress(const void*   inData, size_t   inDataSize,
                                                            void*& outData, size_t& outDataSize, int*& stat, size_t& outNumSegments)
{
    // Check if null data pointer or size passed into the method
    //
    if( !inData || !inDataSize ) return ErrNoData;

    // Check if the specified segment size is compatible with the compression algorithm
    //
    if( m_segmentSize &&
        COMPRESSOR::ImageParams::MinDepth &&
        m_segmentSize % COMPRESSOR::ImageParams::MinDepth ) return ErrNotSupportedSize;

    // Check if the input data size is compatible with the compression algorithm
    //
    if( COMPRESSOR::ImageParams::MinDepth &&
        inDataSize % COMPRESSOR::ImageParams::MinDepth ) return ErrNotSupportedSize;
        
    // Calculate the number of segments. Reallocate segment arrays if needed.
    //
    // NOTES: If segment size is zero then we have only 1 segment.
    //        Otherwise we try to split the input buffer into segments
    //        of the preffered size with an exception of the very last
    //        one which may get bigger if the tail of the buffer would
    //        be shorted than the preferred segment size.
    //
    size_t numSegments = ( !m_segmentSize || ( inDataSize < m_segmentSize )) ?
                         1 :
                         inDataSize / m_segmentSize; // + ( inDataSize % m_segmentSize ? 1 : 0 );

    if( m_numSegments < numSegments ) {
        m_numSegments = numSegments;

        delete [] m_segmentInData;
        delete [] m_segmentInDataParams;

        delete [] m_segmentOutData;
        delete [] m_segmentOutDataSize;

        delete [] m_segmentStat;

        m_segmentInData       = new                            void*[m_numSegments];
        m_segmentInDataParams = new typename COMPRESSOR::ImageParams[m_numSegments];

        m_segmentOutData     = new  void*[m_numSegments];
        m_segmentOutDataSize = new size_t[m_numSegments];

        m_segmentStat = new int[m_numSegments];
    }

    // Split the input buffer into segments and initiate the compression
    //
    const unsigned char* ptr = (const unsigned char*)inData;
    for( size_t s = 0; s < numSegments - 1; ++s, ptr += m_segmentSize ) {
        m_segmentInData      [s]        = (void*)ptr;
        m_segmentInDataParams[s].width  = m_segmentSize / COMPRESSOR::ImageParams::MinDepth;
        m_segmentInDataParams[s].height = 1;
        m_segmentInDataParams[s].depth  = COMPRESSOR::ImageParams::MinDepth;
    }
    m_segmentInData      [numSegments - 1]        = (void*)ptr;
    m_segmentInDataParams[numSegments - 1].width  = (((const unsigned char*)inData + inDataSize) - ptr) / COMPRESSOR::ImageParams::MinDepth;
    m_segmentInDataParams[numSegments - 1].height = 1;
    m_segmentInDataParams[numSegments - 1].depth  = COMPRESSOR::ImageParams::MinDepth;

#ifdef PDS_CODEC_STREAMCOMPRESSOR_DEBUG
    std::cout << "Pds::Codec::StreamCompressor<COMPRESSOR,MT>::compress()::DEBUG {\n"
        << "         inData: " << std::setw(20) << std::setfill('0') << (unsigned long long)inData << "\n"
        << "     inDataSize: " << inDataSize  << "\n"
        << "    numSegments: " << numSegments << "\n"
        << "     in segment  address               size\n";
    for( size_t s = 0; s < numSegments ; ++s ) {
        std::cout
        << "        " << std::setw(7) << std::setfill(' ') << s
        << "  " << std::setw(20) << std::setfill('0') << (unsigned long long)m_segmentInData[s]
        << "  " << m_segmentInDataParams[s].width * m_segmentInDataParams[s].height * m_segmentInDataParams[s].depth << "\n";
    }
#endif

    int status = m_compressor->compress((const void**)m_segmentInData,  m_segmentInDataParams,
                                         m_segmentOutData, m_segmentOutDataSize, m_segmentStat, numSegments );
    if( status ) {
        if( status ==  MT<COMPRESSOR >::ErrInAlgorithm ) {
            status = ErrInCompressor;  // Specific details for each segment can be found in the array
                                       // of status values.
        } else {
            status = ErrInMT;
            m_segmentStat[0] = status;
        }
        stat = m_segmentStat;
    } else {

        // Calculate the total size of the compressed buffer, including its header:
        //
        //   32-bit: number of segments
        //   32-bit: 1st segment size (bytes)
        //   ..
        //   32-bit: last segment size (bytes)
        //   <1st segment>
        //   ..
        //   <last segment>
        //
        outDataSize = sizeof(uint32_t)  + sizeof(uint32_t) * numSegments;
#ifdef PDS_CODEC_STREAMCOMPRESSOR_DEBUG
        const size_t outDataHdrSize = outDataSize;
#endif
        for( size_t s = 0; s < numSegments; ++s ) outDataSize += m_segmentOutDataSize[s];
 
        // Reallocate the cache size if needed.
        //
        if( outDataSize > m_outDataSize ) {
            m_outDataSize = outDataSize;
            delete [] (uint8_t*)m_outData;
            m_outData = (void*)(new uint8_t[m_outDataSize]);
        }

        // Fill in the header
        //
        uint8_t* ptr = (uint8_t*)m_outData;
        *(uint32_t*)ptr = numSegments; ptr += sizeof(uint32_t);
        for( size_t s = 0; s < numSegments; ++s ) {
            *(uint32_t*)ptr = m_segmentOutDataSize[s]; ptr += sizeof(uint32_t);
        }

        // Copy over the segments
        //
        for( size_t s = 0; s < numSegments; ++s ) {
            memcpy((void*)ptr, m_segmentOutData[s], m_segmentOutDataSize[s]); ptr += m_segmentOutDataSize[s];
        }

        outData = m_outData;
        outNumSegments = numSegments;

#ifdef PDS_CODEC_STREAMCOMPRESSOR_DEBUG
        std::cout
            << "        outData: " << std::setw(20) << std::setfill('0') << (unsigned long long)outData << "\n"
            << "    outDataSize: " << outDataSize  << "\n"
            << " outDataHdrSize: " << outDataHdrSize << "\n"
            << "    out segment  size     compression\n";
        for( size_t s = 0; s < numSegments ; ++s ) {
            std::cout
            << "        " << std::setw(7) << std::setfill(' ') << s
            << "  " << m_segmentOutDataSize[s]
            << "  " << 1.0 * m_segmentOutDataSize[s] / ( m_segmentInDataParams[s].width * m_segmentInDataParams[s].height * m_segmentInDataParams[s].depth ) << "\n";
        }
        COMPRESSOR debug_compressor;
        for( size_t s = 0; s < numSegments ; ++s ) {
            std::cout
            << "    out segment: " << s << "\n";
                debug_compressor.dump(std::cout, (const void*)m_segmentOutData[s], m_segmentOutDataSize[s] );
        }
#endif
    }
#ifdef PDS_CODEC_STREAMCOMPRESSOR_DEBUG
    std::cout
         << "}\n";
#endif
    return status;
}

template< class COMPRESSOR, template< class COMPRESSOR > class MT >
int
Pds::Codec::StreamCompressor<COMPRESSOR,MT>::decompress(const void* outData, const size_t outDataSize,
                                                              void*& inData,       size_t& inDataSize, int*& stat, size_t& inNumSegments)
{
    // Check if null data pointer or size passed into the method
    //
    if( !outData || !outDataSize ) return ErrNoData;

    // Unpack and verify the header
    //
    //   32-bit: number of segments
    //   32-bit: 1st segment size (bytes)
    //   ..
    //   32-bit: last segment size (bytes)
    //   <1st segment>
    //   ..
    //   <last segment>
    //
    uint8_t* ptr = (uint8_t*)outData;
    if( outDataSize < sizeof(uint32_t))                                       return ErrBadFormat;  // not enough room for the number of segments
    const size_t numSegments = *(uint32_t*)ptr; ptr += sizeof(uint32_t);

    if( !numSegments )                                                        return ErrBadFormat;  // there must be at least one segment
    const size_t hdrSize = sizeof(uint32_t) + numSegments * sizeof(uint32_t);
    if( outDataSize < hdrSize )                                               return ErrBadFormat;  // not enough room for segment sizes

    size_t dataSize = hdrSize;
    for( size_t s = 0; s < numSegments; ++s ) {
        const size_t segmentSize = *(uint32_t*)ptr; ptr += sizeof(uint32_t);
        if( outDataSize < segmentSize )                                       return ErrBadFormat;  // this segment is too large
        dataSize += segmentSize;
        if( outDataSize < dataSize )                                          return ErrBadFormat;  // not enough room for this segment
    }
    if( outDataSize != dataSize )                                             return ErrBadFormat;  // buffer size doesn't match its structure encoded in the header

    // Reallocate segment arrays if needed
    //
    if( m_numSegments < numSegments ) {
        m_numSegments = numSegments;

        delete [] m_segmentInData;
        delete [] m_segmentInDataParams;

        delete [] m_segmentOutData;
        delete [] m_segmentOutDataSize;

        delete [] m_segmentStat;

        m_segmentInData       = new                            void*[m_numSegments];
        m_segmentInDataParams = new typename COMPRESSOR::ImageParams[m_numSegments];

        m_segmentOutData     = new  void*[m_numSegments];
        m_segmentOutDataSize = new size_t[m_numSegments];

        m_segmentStat = new int[m_numSegments];
    }

    // Prepare pointers to compressed segments and their sizes
    //
    uint8_t* segment_size_ptr = (uint8_t*)outData + sizeof(uint32_t);               // skip the number of seconds
    uint8_t* segment_ptr      = segment_size_ptr  + sizeof(uint32_t) * numSegments; // skip the rest of the header

    for( size_t s = 0; s < numSegments; ++s ) {
        const size_t segmentSize = *(uint32_t*)segment_size_ptr; segment_size_ptr += sizeof(uint32_t);
        m_segmentOutData    [s]  = (void*)segment_ptr; segment_ptr += segmentSize;
        m_segmentOutDataSize[s]  = segmentSize;
    }

#ifdef PDS_CODEC_STREAMCOMPRESSOR_DEBUG
    std::cout << "Pds::Codec::StreamCompressor<COMPRESSOR,MT>::decompress()::DEBUG {\n"
        << "        outData: " << std::setw(20) << std::setfill('0') << (unsigned long long)outData << "\n"
        << "        hdrSize: " << hdrSize << "\n"
        << "    outDataSize: " << outDataSize  << "\n"
        << "    numSegments: " << numSegments << "\n"
        << "    out segment  address               size\n";
    for( size_t s = 0; s < numSegments ; ++s ) {
        std::cout
        << "        " << std::setw(7) << std::setfill(' ') << s
        << "  " << std::setw(20) << std::setfill('0') << (unsigned long long)m_segmentOutData[s]
        << "  " << m_segmentOutDataSize[s] << "\n";
    }
#endif
    // Decompress the segments and analyze status code
    //
    int status = m_compressor->decompress((const void**)m_segmentOutData, m_segmentOutDataSize,
                                           m_segmentInData,  m_segmentInDataParams, m_segmentStat, numSegments );
    if( status ) {
        if( status ==  MT<COMPRESSOR >::ErrInAlgorithm ) {
            status = ErrInCompressor;  // Specific details for each segment can be found in the array
                                       // of status values.
        } else {
            status = ErrInMT;
            m_segmentStat[0] = status;
        }
        stat = m_segmentStat;
    } else {

        // Calculate the total size of the uncompressed buffer,

        inDataSize = 0;
        for( size_t s = 0; s < numSegments; ++s )
            inDataSize += m_segmentInDataParams[s].width  *
                          m_segmentInDataParams[s].height *
                          m_segmentInDataParams[s].depth;

        // Reallocate the cache size if needed.
        //
        if( inDataSize > m_inDataSize ) {
            m_inDataSize = inDataSize;
            delete [] (uint8_t*)m_inData;
            m_inData = (void*)(new uint8_t[m_inDataSize]);
        }

        // Copy over the uncompressed segments
        //
        uint8_t* ptr = (uint8_t*)m_inData;
        for( size_t s = 0; s < numSegments; ++s ) {
            const size_t segmentSize = m_segmentInDataParams[s].width  *
                                       m_segmentInDataParams[s].height *
                                       m_segmentInDataParams[s].depth;
            memcpy((void*)ptr, m_segmentInData[s], segmentSize); ptr += segmentSize;
        }

        inData = m_inData;
        inNumSegments = numSegments;

#ifdef PDS_CODEC_STREAMCOMPRESSOR_DEBUG
        std::cout
            << "         inData: " << std::setw(20) << std::setfill('0') << (unsigned long long)inData << "\n"
            << "     inDataSize: " << inDataSize  << "\n"
            << "     in segment  size     compression\n";
        for( size_t s = 0; s < numSegments ; ++s ) {
            std::cout
            << "        " << std::setw(7) << std::setfill(' ') << s
            << "  " << ( m_segmentInDataParams[s].width * m_segmentInDataParams[s].height * m_segmentInDataParams[s].depth )
            << "  " << 1.0 * m_segmentOutDataSize[s] / ( m_segmentInDataParams[s].width * m_segmentInDataParams[s].height * m_segmentInDataParams[s].depth ) << "\n";
        }
#endif
    }
#ifdef PDS_CODEC_STREAMCOMPRESSOR_DEBUG
    std::cout
         << "}\n";
#endif

    return status;
}

template< class COMPRESSOR, template< class COMPRESSOR > class MT >
const char*
Pds::Codec::StreamCompressor<COMPRESSOR,MT>::err2str(int code)
{
    switch( code ) {
    case Success:             return "Success";
    case ErrNotSupportedSize: return "Input segment or data size is not supported by the compression algorithm";
    case ErrNoData:           return "No data to process (null data size or null pointer)";
    case ErrBadFormat:        return "Unsupported or bad format of the input buffer";
    case ErrInCompressor:     return "Errors reported by by the compression algorithm";
    case ErrInMT:             return "Errors reported by the multi-threaded wrapper";
    }
    return "unknown error code";
}

#endif  // PDS_CODEC_STREAMCOMPRESSOR_H
