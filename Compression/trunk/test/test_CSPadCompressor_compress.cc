/* This is a simple implementation of the run-length encoding
 * algorithm.
 */

#include "Compression/CSPadCompressor.hh"

using namespace Pds::Codec;

#include <iostream>

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

namespace {

    const size_t DEFAULT_RECLEN = 71780;

    class CompressionTest{

    public:

        CompressionTest(size_t reclen, unsigned int num_iter_per_image) :
            m_compressor        (new CSPadCompressor()),
            m_num_iter_per_image(num_iter_per_image),
            m_inbufsize         (reclen),
            m_inbuf             (new unsigned short[reclen] )
        {}

        virtual ~CompressionTest()
        {
            delete m_compressor;
            delete [] m_inbuf;
        }

        bool run( const char* infilename, const char* outfilename, bool stats, bool test, bool verbose )
        {
            FILE* infile = fopen64( infilename, "r" );
            if( !infile ) {
                cerr << "failed to open: " << infilename << " due to: " << last_error() << endl;
                return false;
            }
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

            void* outData  = 0;
            size_t outDataSize = 0;

            size_t num_read   = 0;
            while( 0 != ( num_read = fread( m_inbuf, sizeof(unsigned short), m_inbufsize, infile ))) {

                CSPadCompressor::ImageParams params;
                params.width  = num_read;
                params.height = 1;
                params.depth  = 2;

                for( unsigned int i = 0; i < m_num_iter_per_image; ++i ) {
                    const int stat = m_compressor->compress( m_inbuf, params, outData, outDataSize );
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
                    float compression =  1.0 * outDataSize / ( sizeof(unsigned short) * num_read );
                    printf( "image compression: %f\n", compression );

                    const size_t hdr_size_bytes =
                        sizeof(unsigned int)    +  // compression flags
                        sizeof(unsigned int)    +  // original image checksum 
                        sizeof(unsigned int)    +  // width
                        sizeof(unsigned int)    +  // height
                        sizeof(unsigned int)    +  // depth
                        sizeof(unsigned int)    ;  // compressed data block size

                    if( outDataSize < hdr_size_bytes ) {
                        cerr << "compression failed: the total compressed size can't be less than " << hdr_size_bytes << " bytes" << endl;
                        return false;
                    }
                    unsigned char* hdr_ptr = (unsigned char*)outData;

                    const unsigned int hdr_compression_flag      = *((unsigned int*)hdr_ptr); hdr_ptr += sizeof(unsigned int);
                    const unsigned int hdr_original_checksum     = *((unsigned int*)hdr_ptr); hdr_ptr += sizeof(unsigned int); 

                    CSPadCompressor::ImageParams params;
                    params.width                                 = *((unsigned int*)hdr_ptr); hdr_ptr += sizeof(unsigned int);
                    params.height                                = *((unsigned int*)hdr_ptr); hdr_ptr += sizeof(unsigned int);
                    params.depth                                 = *((unsigned int*)hdr_ptr); hdr_ptr += sizeof(unsigned int);
                    const unsigned int hdr_compressed_size_bytes = *((unsigned int*)hdr_ptr); hdr_ptr += sizeof(unsigned int);

                    if( hdr_size_bytes + hdr_compressed_size_bytes != outDataSize ) {
                        cerr << "compression failed: inconsistent size of the compressed image and its payload." << endl;
                        return false;
                    }
                    cout << "HEADER {\n"
                         << "  compression flags : " << hdr_compression_flag << "\n"
                         << "  original image cs : " << hdr_original_checksum << "\n"
                         << "  ImageParams.width : " << params.width << "\n"
                         << "             .height: " << params.height << "\n"
                         << "             .depth : " << params.depth << "\n"
                         << "  compressed size   : " << hdr_compressed_size_bytes << " Bytes\n"
                         << "}" << endl;

                    if( 0 == hdr_compression_flag ) {

                        if( params.width * params.height * sizeof(unsigned short) !=
                            hdr_compressed_size_bytes ) {

                            cerr << "compression failed: inconsistent size of the uncompressed data block" << endl;
                            return false;
                        }
                        unsigned int    cs = 0;
                        unsigned short* ptr_begin = (unsigned short*)hdr_ptr;
                        unsigned short* ptr_end   = ptr_begin + hdr_compressed_size_bytes / sizeof(unsigned short);
                        for( unsigned short* ptr  = ptr_begin;
                                             ptr != ptr_end;
                                           ++ptr ) cs += *ptr;
                        if( hdr_original_checksum != cs ) {
                            cerr << "compression failed: checksum doesn't match the original one" << endl;
                            return false;
                        }
                    } else {

                        unsigned char* data_ptr = hdr_ptr;

                        const unsigned short base            = *((unsigned short*)data_ptr); data_ptr += sizeof(unsigned short);
                        const size_t         data_size_bytes = *((unsigned int*  )data_ptr); data_ptr += sizeof(unsigned int);

                        if( hdr_compressed_size_bytes <
                            sizeof(unsigned short)    +
                            sizeof(unsigned int)      + data_size_bytes ) {

                            cerr << "compression failed: inconsistent size of the compressed data block" << endl;
                            return false;
                        }
                        unsigned char* bitmap_ptr = data_ptr + data_size_bytes;

                        const size_t bitmap_size_shorts = *((unsigned int*)bitmap_ptr); bitmap_ptr += sizeof(unsigned int);

                        if( hdr_compressed_size_bytes !=
                            sizeof(unsigned short)    +
                            sizeof(unsigned int)      +
                            data_size_bytes           +
                            sizeof(unsigned int)      +
                            sizeof(unsigned short) * bitmap_size_shorts ) {

                            cerr << "compression failed: inconsistent size of the bitmap data block" << endl;
                            return false;
                        }
                        cout << "DATA {\n"
                             << "  base               : " << base << "\n"
                             << "  data size bytes    : " << data_size_bytes << "\n"
                             << "  bitmap size shorts : " << bitmap_size_shorts << "\n"
                             << "}" << endl;
                    }
                }
                if( stats ) {
                    total_bytes_read       += sizeof(unsigned short) * num_read;
                    total_bytes_compressed += outDataSize;
                }
                if( test )
                    if( !this->test( params, outData, outDataSize ))
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
            bool result = feof( infile );

            if( result )               cerr << "end of file" << endl;
            else if( ferror( infile )) cerr << "failed to read the file: " << infilename << " due to: " << last_error() << endl;
            else                       cerr << "unknown error occured when reading the file:  " << infilename << endl;

            fclose( infile );

            if( outfile ) fclose( outfile );

            return result;
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
        size_t           m_inbufsize;   // the number of 16-bit words in the input buffer
        unsigned short*  m_inbuf;
    };

    void usage( const char* msg=0)
    {
        if( msg ) cerr << msg << "\n";
        cerr << "usage: <infile> [-o <outfile>] [-r <reclen_shorts>] [-i <num_iter_per_image>] [-s] [-t] [-v]" << endl;
    }
}

int
main( int argc, char* argv[] )
{
    int   numArgs = argc - 1;
    char** argsPtr = argv; ++argsPtr;

    if( numArgs <= 0 ) { ::usage(); return 1; }

    const char* infilename  = *(argsPtr++); --numArgs;
    char*       outfilename = 0;
    size_t      reclen      = DEFAULT_RECLEN;
    bool        stats       = false;
    bool        test        = false;
    bool        verbose     = false;
    unsigned int num_iter_per_image = 1;

    while( numArgs ) {

        const char* opt = *(argsPtr++); --numArgs;

        if( !strcmp( opt, "-r" )) {
            if( !numArgs )                                     { ::usage( "record value isn't following the option" );        return 1; }
            const char* val = *(argsPtr++); --numArgs;
            if( 1 != sscanf( val, "%lu", &reclen ))             { ::usage( "failed to translate a value of <reclen_shorts>" ); return 1; }
            if( reclen == 0 )                                  { ::usage( "<reclen_shorts> parameter can't be 0" );           return 1; }

        } else if( !strcmp( opt, "-i" )) {
            if( !numArgs )                                     { ::usage( "iteration number value isn't following the option" );        return 1; }
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

    return ct.run( infilename, outfilename, stats, test, verbose ) ? 0 : 1;
}
