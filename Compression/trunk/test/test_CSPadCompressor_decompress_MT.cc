/* This is a test for the CSPad decompression algorithm.
 */

#include "Compression/CSPadCompressor.hh"
#include "Compression/CompressorMT.hh"

using namespace Pds::Codec;

#include <iostream>

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

namespace {

    const size_t MAX_IMAGE_SIZE  = 16*1024*1024;

    class DecompressionTestMT {

    public:

        DecompressionTestMT(size_t       num_images_per_batch,
                            unsigned int num_iter_per_batch,
                            size_t       maxImagesPerThread,
                            size_t       numThreads) :

            m_num_images_per_batch(num_images_per_batch),
            m_num_iter_per_batch  (num_iter_per_batch),
            m_compressor          (new CompressorMT<CSPadCompressor>(maxImagesPerThread, numThreads))

        {
            m_inbufsize = new size_t[num_images_per_batch];
            m_inbuf = new uint8_t*[num_images_per_batch];
            for( size_t i = 0; i < m_num_images_per_batch; ++i ) {
                m_inbufsize[i] = 0;
                m_inbuf[i] = new uint8_t[MAX_IMAGE_SIZE];
            }
        }

        virtual ~DecompressionTestMT()
        {
            delete m_compressor;
            m_compressor = 0;
            delete [] m_inbufsize;
            m_inbufsize = 0;
            for( size_t i = 0; i < m_num_images_per_batch; ++i )
                delete [] m_inbuf[i];
            delete [] m_inbuf;
            m_inbuf = 0;
        }

        bool run( const char* infilename, const char* outfilename, bool stats, bool dump, bool verbose, bool no_decompression )
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
            unsigned long long total_bytes_decompressed = 0ULL;

            void**  image       = new void* [m_num_images_per_batch];
            size_t* outDataSize = new size_t[m_num_images_per_batch];
            int*    stat4all    = new int   [m_num_images_per_batch];

            CSPadCompressor::ImageParams* params = new CSPadCompressor::ImageParams[m_num_images_per_batch];

            while( true ) {

                size_t num_images = 0;
                for( size_t i = 0; i < m_num_images_per_batch; ++i ) {

                    // Read a compressed image header first in order to determine
                    // a size of the image body.
                    //
                    // NOTE: We're limitting the maximum size of compressed images
                    //       to prevent application hang when reading wrong/corrupt
                    //       files.
                    //
                    const size_t hdr_size =
                        sizeof(uint32_t)     +  // compression fgals
                        sizeof(uint32_t)     +  // original image checksum
                        sizeof(uint32_t) * 3 +  // image params
                        sizeof(uint32_t);       // body size (original image or compressed image + bitmap)
 
                    uint8_t* ptr = m_inbuf[i];

                    if( hdr_size != fread( ptr, sizeof(uint8_t), hdr_size, infile )) break;  // either EOF or not enough data for a full header

                    const uint32_t compression_flags = *(uint32_t*)ptr; ptr += sizeof(uint32_t);
                    const uint32_t cs                = *(uint32_t*)ptr; ptr += sizeof(uint32_t);
                    const uint32_t width             = *(uint32_t*)ptr; ptr += sizeof(uint32_t);
                    const uint32_t height            = *(uint32_t*)ptr; ptr += sizeof(uint32_t);
                    const uint32_t depth             = *(uint32_t*)ptr; ptr += sizeof(uint32_t);
                    const uint32_t body_size         = *(uint32_t*)ptr; ptr += sizeof(uint32_t);

                    if( 0 == body_size ) {
                        cerr << "error in image header: image body can't have 0 length. The file may be wrong or corrupt." << endl;
                        return false;
                    }
                    if( body_size > MAX_IMAGE_SIZE - hdr_size ) {
                        cerr << "error in image header: image body can't be longer than " << MAX_IMAGE_SIZE - hdr_size
                             << " bytes. The file may be wrong or corrupt." << endl;
                        return false;
                    }

                    // Read the rest of the image
                    //
                    if( body_size != fread( ptr, sizeof(uint8_t), body_size, infile )) {
                        cerr << "error while reading the image body. The file may be wrong or corrupt." << endl;
                        return false;
                    }
                    m_inbufsize[i] = hdr_size + body_size;

                    ++num_images;
                }
                if( 0 == num_images ) break;

                for( unsigned int iter = 0; iter < m_num_iter_per_batch; ++iter ) {
                    if( no_decompression ) {

                        // Skip the decompression

                        if( stats ) {
                            for( size_t i = 0; i < num_images; ++i ) {
                                total_bytes_read        += m_inbufsize[i];
                                total_bytes_decompressed = total_bytes_read;
                            }
                        }

                    } else {
                        const int stat =
                            m_compressor->decompress(
                                (const void**)m_inbuf,
                                m_inbufsize,
                                image,
                                params,
                                stat4all,
                                num_images
                            );
                        if( stat ) {
                            cerr << "de compression failed with error status: " << stat << "\n";
                            for( size_t i = 0; i < num_images; ++i )
                                cerr << "  stat4all[" << i << "]: " << stat4all[i] << "\n";
                            cerr << endl;
                            return false;
                        }
                        if( stats ) {
                            for( size_t i = 0; i < num_images; ++i ) {
                                total_bytes_read         += m_inbufsize[i];
                                total_bytes_decompressed += sizeof(unsigned short) * params[i].width;
                            }
                        }
                    }
                }
                if( outfile ) {
                    for( size_t i = 0; i < num_images; ++i ) {
                        if( sizeof(unsigned short) * params[i].width != fwrite( image[i], sizeof(uint8_t), sizeof(unsigned short) * params[i].width, outfile )) {
                            if( ferror( outfile )) cerr << "failed to write into the output file due to: " << last_error() << endl;
                            else                   cerr << "unknown error when writing into the output file" << endl;
                                return false;
                        }
                    }
                }

                if( verbose ) {
                    for( size_t i = 0; i < num_images; ++i ) {
                        float compression =  1.0 * m_inbufsize[i] / ( sizeof(unsigned short) * params[i].width );
                        printf( "detected image compression: %f\n", compression );
                    }
                }
                if( dump ) {
                    CSPadCompressor compressor;
                    for( size_t i = 0; i < num_images; ++i ) {
                        cout << "\n";
                        compressor.dump( cout, m_inbuf[i], m_inbufsize[i] );
                    }
                }
            }
            delete [] stat4all;
            delete [] params;

            struct timespec end;
            clock_gettime( CLOCK_REALTIME, &end );

            if( stats ) {
                float compression =  1.0 * total_bytes_read / total_bytes_decompressed;
                printf( "detected average compression: %f\n", compression );

                const double secs = end.tv_sec + end.tv_nsec/1e9 - begin.tv_sec - begin.tv_nsec/1e9;
                const double mbps = total_bytes_read / 1024.0 / 1024.0 / secs;
                printf( "performance: %g MB/s\n", mbps );
            }
            bool result = feof( infile );

            if( result )               cout << "end of file" << endl;
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

    private:

        size_t*   m_inbufsize;  // the number of bytes read into each buffer
        uint8_t** m_inbuf;      // the buffer storage for a batch of compressed images read from an input file

        size_t m_num_images_per_batch;
        size_t m_num_iter_per_batch;

        CompressorMT<CSPadCompressor>* m_compressor;
    };

    void usage( const char* msg=0)
    {
        if( msg ) cerr << msg << "\n";
        cerr << "usage: <infile> [-o <outfile>]\n"
             << "       [-b <num_images_per_batch>]\n"
             << "       [-i <num_iter_per_batch>]\n"
             << "       [-m <max_images_per_thread>\n"
             << "       [-p <max_threads>]\n"
             << "       [-n]\n"
             << "       [-s]\n"
             << "       [-d]\n"
             << "       [-v]" << endl;
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

    size_t       num_images_per_batch = 1;
    unsigned int num_iter_per_batch   = 1;
    size_t       maxImagesPerThread   = 1;
    size_t       numThreads           = 1;

    bool no_decompression = false,
                    stats = false,
                     dump = false,
                  verbose = false;


    while( numArgs ) {

        const char* opt = *(argsPtr++); --numArgs;

        if( !strcmp( opt, "-b" )) {
            if( !numArgs )                                       { ::usage( "missing value for option <num_images_per_batch>" );        return 1; }
            const char* val = *(argsPtr++); --numArgs;
            if( 1 != sscanf( val, "%u", &num_images_per_batch )) { ::usage( "failed to translate a value of <num_images_per_batch>" );  return 1; }
            if( num_images_per_batch == 0 )                      { ::usage( "<num_images_per_batch> can't have a value of 0" );         return 1; }

        } else if( !strcmp( opt, "-i" )) {
            if( !numArgs )                                       { ::usage( "missing value for option <num_iter_per_batch>" );          return 1; }
            const char* val = *(argsPtr++); --numArgs;
            if( 1 != sscanf( val, "%u", &num_iter_per_batch ))   { ::usage( "failed to translate a value of <num_iter_per_batch>" );    return 1; }
            if( num_iter_per_batch == 0 )                        { ::usage( "<num_iter_per_batch> can't have a value of 0" );           return 1; }

        } else if( !strcmp( opt, "-m" )) {
            if( !numArgs )                                       { ::usage( "missing value for option <max_images_per_thread>" );       return 1; }
            const char* val = *(argsPtr++); --numArgs;
            if( 1 != sscanf( val, "%u", &maxImagesPerThread ))   { ::usage( "failed to translate a value of <max_images_per_thread>" ); return 1; }
            if( maxImagesPerThread == 0 )                        { ::usage( "<max_images_per_thread> can't have a value of 0" );        return 1; }

        } else if( !strcmp( opt, "-p" )) {
            if( !numArgs )                                       { ::usage( "missing value for option <max_threads>" );                 return 1; }
            const char* val = *(argsPtr++); --numArgs;
            if( 1 != sscanf( val, "%u", &numThreads ))           { ::usage( "failed to translate a value of <max_threads>" );           return 1; }
            if( numThreads == 0 )                                { ::usage( "<max_threads> can't have a value of 0" );                  return 1; }

        } else if( !strcmp( opt, "-o" )) {
            if( !numArgs )                                       { ::usage( "record value isn't following the option" );                return 1; }
            outfilename = *(argsPtr++); --numArgs;

        } else if( !strcmp( opt, "-n" )) {
            no_decompression = true;

        } else if( !strcmp( opt, "-s" )) {
            stats = true;

        } else if( !strcmp( opt, "-d" )) {
            dump = true;

        } else if( !strcmp( opt, "-v" )) {
            verbose = true;

        } else                                                 { ::usage( "unknown command option" );                          return 1; }
    }
    if( numArgs )                                              { ::usage( "illegal number of parameters" );                    return 1; }

    printf( "maximum compressed image size (bytes): %u\n", MAX_IMAGE_SIZE );
    printf( "images per batch:                      %u\n", num_images_per_batch );
    printf( "iterations per batch:                  %u\n", num_iter_per_batch );
    printf( "maximum number of images per thread:   %u\n", maxImagesPerThread );
    printf( "maximum number of threads:             %u\n", numThreads );

    ::DecompressionTestMT dct( num_images_per_batch, num_iter_per_batch, maxImagesPerThread, numThreads );

    return dct.run( infilename, outfilename, stats, dump, verbose, no_decompression ) ? 0 : 1;
}
