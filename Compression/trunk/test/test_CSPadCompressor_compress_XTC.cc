/* This is a simple implementation of the run-length encoding
 * algorithm.
 */

#include "Compression/CSPadCompressor.hh"
#include "Compression/CompressorMT.hh"
#include "Compression/StreamCompressor.hh"

#include "XtcInput/XtcDgIterator.h"

using namespace Pds::Codec;

#include <iostream>
#include <iomanip>

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

namespace {

    class CompressionTest{

    public:

        CompressionTest( size_t       reclen,
                         unsigned int num_threads,
                         unsigned int num_iter_per_image ) :

            m_compressor        (new StreamCompressor<CSPadCompressor,CompressorMT>( reclen, num_threads )),
            m_num_iter_per_image(num_iter_per_image)
        {}

        virtual ~CompressionTest()
        {
            delete m_compressor;
        }

        bool run( const char*  infilename,
                  const char*  outfilename,
                  unsigned int first_dgram_num,
                  bool         stats,
                  bool         test,
                  bool         verbose )
        {
            const size_t maxDgramSize = 32*1024*1024;
            XtcInput::XtcDgIterator itr( infilename, maxDgramSize );

            FILE* outfile = 0;
            if( outfilename ) {
                outfile = fopen64( outfilename, "w" );
                if( !outfile ) {
                    cerr << "failed to open: " << outfilename << " due to: " << last_error() << endl;
                    return false;
                }
            }

            struct timespec begin;
            clock_gettime( CLOCK_REALTIME, &begin );

            unsigned long long total_bytes_read = 0ULL;
            unsigned long long total_bytes_compressed = 0ULL;

            unsigned int dgram_num = 0;

            while(true) {

                XtcInput::Dgram::ptr dgram = itr.next();
                if( !dgram.get()) break;

                if( dgram_num++ < first_dgram_num ) continue;

                const void* inData     = (const void*)(dgram->xtc.payload());
                size_t      inDataSize = dgram->xtc.sizeofPayload();

                if( verbose ) {
                    cout << "\n"
                         << "************************************************************************\n"
                         << "*************************** DATAGRAM: " << setw(6) << setfill(' ') << (dgram_num - 1) << " ***************************\n"
                         << "************************************************************************\n"
                         << "\n"
                         << "inDataSize: " << inDataSize << " Bytes\n";
                }
                
                void*  outData        = 0;
                size_t outDataSize    = 0;
                int*   outSegmentStat = 0;
                size_t outNumSegments = 0;

                for( unsigned int i = 0; i < m_num_iter_per_image; ++i ) {
                    const int status = m_compressor->compress( inData, inDataSize,
                                                               outData, outDataSize, outSegmentStat, outNumSegments );
                    if( status ) {
                        cerr << "compression failed with error status " << status << ": " << m_compressor->err2str( status ) << endl;
                        return false;
                    }
                }
                if( outfile ) {
                    if( outDataSize != fwrite( outData, sizeof(unsigned char), outDataSize, outfile )) {
                        if( ferror( outfile )) cerr << "failed to write into the output file due to: " << last_error() << endl;
                        else                   cerr << "unknown error when writing into the output file" << endl;
                            return false;
                    }
                }
                if( verbose ) {
                    double compression =  1.0 * outDataSize / inDataSize;
                    cout << setprecision(4) << "image compression: " << compression << "  (" << outDataSize << "/" << inDataSize << ")\n";
                }
                if( stats ) {
                    total_bytes_read       += inDataSize;
                    total_bytes_compressed += outDataSize;
                }
                if( test )
                    if( !this->test( inData, inDataSize, outData, outDataSize, outNumSegments ))
                        return false;
            }

            struct timespec end;
            clock_gettime( CLOCK_REALTIME, &end );

            if( stats ) {
                float compression =  1.0 * total_bytes_compressed / total_bytes_read;
                printf( "average compression: %f\n", compression );

                const double secs = end.tv_sec + end.tv_nsec/1e9 - begin.tv_sec - begin.tv_nsec/1e9;
                const double mbps = m_num_iter_per_image * total_bytes_read / 1024.0 / 1024.0 / secs;
                printf( "performance: %g MB/s\n", mbps );
            }

            if( outfile ) fclose( outfile );

            return true;
        }

    private:

        string last_error() const
        {
            char errbuf[256];
            if( !strerror_r( errno, errbuf, 256 )) return errbuf;
            return "failed to obtain error information";
        }

        bool test( const void*  inDataExpected, size_t  inDataSizeExpected,
                         void* outData,         size_t outDataSize,         size_t outNumSegments )
        {
            void*  inData        = 0;
            size_t inDataSize    = 0;
            int*   inSegmentStat = 0;
            size_t inNumSegments = 0;

            const int status = m_compressor->decompress( outData, outDataSize,
                                                          inData,  inDataSize, inSegmentStat, inNumSegments );
            if( status ) {
                cerr << "decompression failed with error status " << status << ": " << m_compressor->err2str( status ) << endl;
                return false;
            }
            if( inDataSizeExpected != inDataSize ) {
                cerr << "test failed: data size missmatch after decompression, got " << inDataSize << " instead of " << inDataSizeExpected << endl;
                return false;
            }
            if( outNumSegments != inNumSegments ) {
                cerr << "test failed: number of segments missmatch after decompression, got " << inNumSegments << " instead of " << outNumSegments << endl;
                return false;
            }
            for( size_t i = 0; i < inDataSizeExpected; ++i ) {
                if(((char*)inDataExpected)[i] != ((char*)inData)[i] ) {
                    cerr << "test failed: data missmatch at byte position " << i << " after decompression, got " << ((char*)inData)[i] << " instead of " <<  ((char*)inDataExpected)[i]  << endl;
                    return false;
                }
            }
            return true;
        }

    private:

        StreamCompressor<CSPadCompressor,CompressorMT>* m_compressor;
        unsigned int m_num_iter_per_image;
    };

    void usage( const char* msg=0)
    {
        if( msg ) cerr << msg << "\n";
        cerr << "usage: <infile> [-o <outfile>] [-f <first_dgram_num>] [-r <reclen_shorts>] [-p <num_threads>] [-i <num_iter_per_image>] [-s] [-t] [-v]" << endl;
    }
}

int
main( int argc, char* argv[] )
{
    int   numArgs = argc - 1;
    char** argsPtr = argv; ++argsPtr;

    if( numArgs <= 0 ) { ::usage(); return 1; }

    const char*  infilename         = *(argsPtr++); --numArgs;
    char*        outfilename        = 0;
    unsigned int first_dgram_num    = 0;
    size_t       reclen             = 0;
    bool         stats              = false;
    bool         test               = false;
    bool         verbose            = false;
    unsigned int num_threads        = 0;
    unsigned int num_iter_per_image = 1;

    while( numArgs ) {

        const char* opt = *(argsPtr++); --numArgs;

        if( !strcmp( opt, "-f" )) {
            if( !numArgs )                                     { ::usage( "datagram index value isn't following the option" );  return 1; }
            const char* val = *(argsPtr++); --numArgs;
            if( 1 != sscanf( val, "%u", &first_dgram_num ))   { ::usage( "failed to translate a value of <first_dgram_num>" ); return 1; }

        } else  if( !strcmp( opt, "-r" )) {
            if( !numArgs )                                     { ::usage( "record value isn't following the option" );        return 1; }
            const char* val = *(argsPtr++); --numArgs;
            if( 1 != sscanf( val, "%lu", &reclen ))            { ::usage( "failed to translate a value of <reclen_shorts>" ); return 1; }
            if( reclen == 0 )                                  { ::usage( "<reclen_shorts> parameter can't be 0" );           return 1; }
            reclen *= sizeof(unsigned short);

        } else if( !strcmp( opt, "-p" )) {
            if( !numArgs )                                     { ::usage( "number of threads value isn't following the option" );   return 1; }
            const char* val = *(argsPtr++); --numArgs;
            if( 1 != sscanf( val, "%u", &num_threads ))        { ::usage( "failed to translate a value of <num_threads>" );         return 1; }

        } else if( !strcmp( opt, "-i" )) {
            if( !numArgs )                                     { ::usage( "iteration number value isn't following the option" );   return 1; }
            const char* val = *(argsPtr++); --numArgs;
            if( 1 != sscanf( val, "%u", &num_iter_per_image )) { ::usage( "failed to translate a value of <num_iter_per_image>" ); return 1; }
            if( num_iter_per_image == 0 )                      { ::usage( "<num_iter_per_image> parameter can't be 0" );           return 1; }

        } else if( !strcmp( opt, "-o" )) {
            if( !numArgs )                                     { ::usage( "record value isn't following the option" );        return 1; }
            outfilename = *(argsPtr++); --numArgs;

        } else if( !strcmp( opt, "-s" )) {
            stats = true;

        } else if( !strcmp( opt, "-t" )) {
            test = true;

        } else if( !strcmp( opt, "-v" )) {
            verbose = true;

        } else                                                 { ::usage( "unknown command option" );                          return 1; }
    }
    if( numArgs )                                              { ::usage( "illegal number of parameters" );                    return 1; }

    ::CompressionTest ct( reclen, num_threads, num_iter_per_image );

    return ct.run( infilename, outfilename, first_dgram_num, stats, test, verbose ) ? 0 : 1;
}
