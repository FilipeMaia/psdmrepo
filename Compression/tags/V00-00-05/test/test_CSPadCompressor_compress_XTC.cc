/* This is a simple implementation of the run-length encoding
 * algorithm.
 */

#include "Compression/CSPadCompressor.hh"
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
                         unsigned int num_iter_per_image ) :

            m_compressor        (new CSPadCompressor()),
            m_num_iter_per_image(num_iter_per_image),
            m_reclen            (reclen)
        {}

        virtual ~CompressionTest()
        {
            delete m_compressor;
        }

        bool run( const char*  infilename,
                  const char*  outfilename,
                  unsigned int first_dgram_num,
                  bool         shift_by_byte,
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

                char* payload = dgram->xtc.payload();
                int payload_size = dgram->xtc.sizeofPayload();

                if( verbose ) {
                    cout << "\n"
                         << "************************************************************************\n"
                         << "*************************** DATAGRAM: " << setw(6) << setfill(' ') << (dgram_num - 1) << " ***************************\n"
                         << "************************************************************************\n"
                         << "\n"
                         << "payload_size: " << payload_size << " Bytes\n";
                }

                size_t num_left = payload_size / sizeof(unsigned short) - ( shift_by_byte ? 1 : 0 );
                unsigned short* inbuf = (unsigned short*)( shift_by_byte ? payload + 1 : payload );

                unsigned int segment_num = 0;

                while( num_left ) {


                    // Use the specified segment size if it's not zero and if there is enough
                    // data left for at least 2 segment. Otherwise compress what ever is left.
                    //
                    size_t num_read = m_reclen && ( num_left / m_reclen >= 2 ) ? m_reclen : num_left;

                    if( verbose ) {
                        cout << "\n"
                             << "*************************** SEGMENT:  " << setw(6) << setfill(' ') << (segment_num++) << " ***************************\n"
                             << "\n"
                             << "segment_size: " << num_read * sizeof(unsigned short) << " Bytes\n";
                    }

                    CSPadCompressor::ImageParams params;
                    params.width  = num_read;
                    params.height = 1;
                    params.depth  = 2;

                    void* outData  = 0;
                    size_t outDataSize = 0;

                    for( unsigned int i = 0; i < m_num_iter_per_image; ++i ) {
                        const int stat = m_compressor->compress( inbuf, params, outData, outDataSize );
                        if( stat ) {
                            cerr << "compression failed with erro status: " << stat << endl;
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
                        double compression =  1.0 * outDataSize / ( sizeof(unsigned short) * num_read );
                        cout << setprecision(4) << "image compression: " << compression << "  (" << outDataSize << "/" << ( sizeof(unsigned short) * num_read ) << ")\n";
                        m_compressor->dump( cout, outData, outDataSize );
                    }
                    if( stats ) {
                        total_bytes_read       += sizeof(unsigned short) * num_read;
                        total_bytes_compressed += outDataSize;
                    }
                    if( test )
                        if( !this->test( params, outData, outDataSize ))
                            return false;

                    // Move to the next segment (if any)
                    //
                    num_left -= num_read;
                    inbuf    += num_read;
                }
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

        bool test( const CSPadCompressor::ImageParams& expected_params, void* outData, size_t outDataSize )
        {
            void* image = 0;
            CSPadCompressor::ImageParams params;
            const int status = m_compressor->decompress( outData, outDataSize, image, params );
            if( 0 != status ) {
                cerr << "test failed: status=" << status << endl;
                return false;
            }
            if(( expected_params.width  != params.width  ) ||
               ( expected_params.height != params.height ) ||
               ( expected_params.depth  != params.depth  )) {
                cerr << "test failed: image parameters missmatch after decompression" << endl;
                return false;
            }
            return true;
        }

    private:

        CSPadCompressor* m_compressor;
        unsigned int     m_num_iter_per_image;

        size_t  m_reclen;  // the preferred segment length (in 16-bit words). If set
                           // to 0 then the input buffer is compressed as a whole w/o
                           // splitting it into segments.
    };

    void usage( const char* msg=0)
    {
        if( msg ) cerr << msg << "\n";
        cerr << "usage: <infile> [-o <outfile>] [-f <first_dgram_num>] [-b] [-r <reclen_shorts>] [-i <num_iter_per_image>] [-s] [-t] [-v]" << endl;
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
    bool         shift_by_byte      = false;
    size_t       reclen             = 0;
    bool         stats              = false;
    bool         test               = false;
    bool         verbose            = false;
    unsigned int num_iter_per_image = 1;

    while( numArgs ) {

        const char* opt = *(argsPtr++); --numArgs;

        if( !strcmp( opt, "-f" )) {
            if( !numArgs )                                     { ::usage( "datagram index value isn't following the option" );  return 1; }
            const char* val = *(argsPtr++); --numArgs;
            if( 1 != sscanf( val, "%u", &first_dgram_num ))   { ::usage( "failed to translate a value of <first_dgram_num>" ); return 1; }

        } else if( !strcmp( opt, "-b" )) {
            shift_by_byte = true;

        } else  if( !strcmp( opt, "-r" )) {
            if( !numArgs )                                     { ::usage( "record value isn't following the option" );        return 1; }
            const char* val = *(argsPtr++); --numArgs;
            if( 1 != sscanf( val, "%lu", &reclen ))            { ::usage( "failed to translate a value of <reclen_shorts>" ); return 1; }
            if( reclen == 0 )                                  { ::usage( "<reclen_shorts> parameter can't be 0" );           return 1; }

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

    printf( "reading records of %lu short numbers\n", reclen );

    ::CompressionTest ct( reclen, num_iter_per_image );

    return ct.run( infilename, outfilename, first_dgram_num, shift_by_byte, stats, test, verbose ) ? 0 : 1;
}
