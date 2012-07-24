/* This is a simple implementation of the run-length encoding
 * algorithm.
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

    const size_t DEFAULT_RECLEN = 71780;

    class CompressionTestMT {

    public:

        CompressionTestMT(size_t       reclen,
                          size_t       num_images_per_batch,
                          unsigned int num_iter_per_batch,
                          size_t       numThreads) :

            m_inbufsize           (reclen),
            m_num_images_per_batch(num_images_per_batch),
            m_num_iter_per_batch  (num_iter_per_batch),
            m_compressor          (new CompressorMT<CSPadCompressor>(numThreads))

        {
            m_inbuf = new unsigned short*[num_images_per_batch];
            for( size_t i = 0; i < m_num_images_per_batch; ++i )
                m_inbuf[i] = new unsigned short[m_inbufsize];
        }

        virtual ~CompressionTestMT()
        {
            delete m_compressor;
            m_compressor = 0;
            for( size_t i = 0; i < m_num_images_per_batch; ++i )
                delete [] m_inbuf[i];
            delete [] m_inbuf;
            m_inbuf = 0;
        }

        bool run( const char* infilename, const char* outfilename, bool stats, bool test, bool dump, bool verbose, bool no_compression )
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

            void**  outData     = new void* [m_num_images_per_batch];
            size_t* outDataSize = new size_t[m_num_images_per_batch];
            int*    stat4all    = new int   [m_num_images_per_batch];

            CSPadCompressor::ImageParams* params = new CSPadCompressor::ImageParams[m_num_images_per_batch];

            while( true ) {

                size_t num_images = 0;
                for( size_t i = 0; i < m_num_images_per_batch; ++i ) {

                    const size_t num_read = fread( m_inbuf[i], sizeof(unsigned short), m_inbufsize, infile );
                    if( 0 == num_read ) break;

                    params[i].width  = num_read;
                    params[i].height = 1;
                    params[i].depth  = 2;

                    ++num_images;
                }
                if( 0 == num_images ) break;

                for( unsigned int iter = 0; iter < m_num_iter_per_batch; ++iter ) {
                    if( no_compression ) {

                        // Skip the compression

                        if( stats ) {
                            for( size_t i = 0; i < num_images; ++i ) {
                                total_bytes_read       += sizeof(unsigned short) * params[i].width;
                                total_bytes_compressed = total_bytes_read;
                            }
                        }

                    } else {
                        const int stat =
                            m_compressor->compress(
                                (const void**)m_inbuf,
                                params,
                                outData,
                                outDataSize,
                                stat4all,
                                num_images
                            );
                        if( stat ) {
                            cerr << "compression failed with error status: " << stat << "\n";
                            for( size_t i = 0; i < num_images; ++i )
                                cerr << "  stat4all[" << i << "]: " << stat4all[i] << "\n";
                            cerr << endl;
                            return false;
                        }
                        if( stats ) {
                            for( size_t i = 0; i < num_images; ++i ) {
                                total_bytes_read       += sizeof(unsigned short) * params[i].width;
                                total_bytes_compressed += outDataSize[i];
                            }
                        }
                    }
                }
                if( outfile ) {
                    for( size_t i = 0; i < num_images; ++i ) {
                        if( outDataSize[i] != fwrite( outData[i], sizeof(unsigned char), outDataSize[i], outfile )) {
                            if( ferror( outfile )) cerr << "failed to write into the output file due to: " << last_error() << endl;
                            else                   cerr << "unknown error when writing into the output file" << endl;
                                return false;
                        }
                    }
                }

                if( verbose ) {
                    for( size_t i = 0; i < num_images; ++i ) {

                        float compression =  1.0 * outDataSize[i] / ( sizeof(unsigned short) * params[i].width );
                        printf( "image compression: %f\n", compression );

                        const size_t hdr_size_bytes =
                            sizeof(unsigned int)    +  // compression flags
                            sizeof(unsigned int)    +  // original image checksum 
                            sizeof(unsigned int)    +  // width
                            sizeof(unsigned int)    +  // height
                            sizeof(unsigned int)    +  // depth
                            sizeof(unsigned int)    ;  // compressed data block size

                        if( outDataSize[i] < hdr_size_bytes ) {
                            cerr << "compression failed: the total compressed size can't be less than " << hdr_size_bytes << " bytes" << endl;
                            return false;
                        }
                        unsigned char* hdr_ptr = (unsigned char*)outData[i];

                        const unsigned int hdr_compression_flag      = *((unsigned int*)hdr_ptr); hdr_ptr += sizeof(unsigned int);
                        const unsigned int hdr_original_checksum     = *((unsigned int*)hdr_ptr); hdr_ptr += sizeof(unsigned int); 

                        CSPadCompressor::ImageParams params;
                        params.width                                 = *((unsigned int*)hdr_ptr); hdr_ptr += sizeof(unsigned int);
                        params.height                                = *((unsigned int*)hdr_ptr); hdr_ptr += sizeof(unsigned int);
                        params.depth                                 = *((unsigned int*)hdr_ptr); hdr_ptr += sizeof(unsigned int);
                        const unsigned int hdr_compressed_size_bytes = *((unsigned int*)hdr_ptr); hdr_ptr += sizeof(unsigned int);

                        if( hdr_size_bytes + hdr_compressed_size_bytes != outDataSize[i] ) {
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
                }
                if( dump ) {
                    CSPadCompressor compressor;
                    for( size_t i = 0; i < num_images; ++i ) {
                        cout << "\n";
                        compressor.dump( cout, outData[i], outDataSize[i] );
                    }
                }
                if( test )
                    if( !this->test( params, (const void**)outData, outDataSize, num_images ))
                        return false;
            }
            delete [] outData;
            delete [] outDataSize;
            delete [] stat4all;
            delete [] params;

            struct timespec end;
            clock_gettime( CLOCK_REALTIME, &end );

            if( stats ) {
                float compression =  1.0 * total_bytes_compressed / total_bytes_read;
                printf( "average compression: %f\n", compression );

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

        bool test( const CSPadCompressor::ImageParams* expected_params,
                   const void** outData,
                   size_t* outDataSize,
                   size_t num_images )
        {
            void** image = new void*[num_images];
            CSPadCompressor::ImageParams* params = new CSPadCompressor::ImageParams[num_images];
            int* stat4all = new int[num_images];

            const int status =
                m_compressor->decompress(
                    outData,
                    outDataSize,
                    image,
                    params,
                    stat4all,
                    num_images
                );
            if( 0 != status ) {

                cerr << "test failed: status=" << status << "\n";
                for( size_t i = 0; i < num_images; ++i )
                    cerr << "  stat4all[" << i << "]: " << stat4all[i] << "\n";
                cerr << endl;
 
                delete [] image;
                delete [] params;
                delete [] stat4all;

                return false;
            }
            for( size_t i = 0; i < num_images; ++i ) {
                if(( expected_params[i].width  != params[i].width  ) ||
                   ( expected_params[i].height != params[i].height ) ||
                   ( expected_params[i].depth  != params[i].depth  )) {

                    cerr << "test failed: image parameters missmatch after decompression of image #" << i << "\n"
                         << "  expected width:  " << expected_params[i].width  << " got: " << params[i].width << "\n"
                         << "           height: " << expected_params[i].height << " got: " << params[i].height << "\n"
                         << "           depth:  " << expected_params[i].depth  << "  got: " << params[i].depth << endl;

                    delete [] image;
                    delete [] params;
                    delete [] stat4all;

                    return false;
                }
            }
            delete [] image;
            delete [] params;
            delete [] stat4all;

            return true;
        }

    private:

        size_t m_inbufsize;             // the number of 16-bit words in an image
        size_t m_num_images_per_batch;
        size_t m_num_iter_per_batch;

        CompressorMT<CSPadCompressor>* m_compressor;

        unsigned short** m_inbuf;      // the buffer storage for a batch of images read from an input file
    };

    void usage( const char* msg=0)
    {
        if( msg ) cerr << msg << "\n";
        cerr << "usage: <infile> [-o <outfile>]\n"
             << "       [-r <reclen_shorts>]\n"
             << "       [-b <num_images_per_batch>]\n"
             << "       [-i <num_iter_per_batch>]\n"
             << "       [-p <max_threads>]\n"
             << "       [-n]\n"
             << "       [-s]\n"
             << "       [-t]\n"
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

    size_t       reclen               = DEFAULT_RECLEN;
    size_t       num_images_per_batch = 1;
    unsigned int num_iter_per_batch   = 1;
    size_t       numThreads           = 1;

    bool no_compression = false;
    bool stats          = false;
    bool test           = false;
    bool dump           = false;
    bool verbose        = false;


    while( numArgs ) {

        const char* opt = *(argsPtr++); --numArgs;

        if( !strcmp( opt, "-r" )) {
            if( !numArgs )                                       { ::usage( "missing value for option <reclen_shorts>" );               return 1; }
            const char* val = *(argsPtr++); --numArgs;
            if( 1 != sscanf( val, "%lu", &reclen ))              { ::usage( "failed to translate a value of <reclen_shorts>" );         return 1; }
            if( reclen == 0 )                                    { ::usage( "<reclen_shorts> can't have a value of 0" );                return 1; }

        } else if( !strcmp( opt, "-b" )) {
            if( !numArgs )                                       { ::usage( "missing value for option <num_images_per_batch>" );        return 1; }
            const char* val = *(argsPtr++); --numArgs;
            if( 1 != sscanf( val, "%lu", &num_images_per_batch )) { ::usage( "failed to translate a value of <num_images_per_batch>" );  return 1; }
            if( num_images_per_batch == 0 )                      { ::usage( "<num_images_per_batch> can't have a value of 0" );         return 1; }

        } else if( !strcmp( opt, "-i" )) {
            if( !numArgs )                                       { ::usage( "missing value for option <num_iter_per_batch>" );          return 1; }
            const char* val = *(argsPtr++); --numArgs;
            if( 1 != sscanf( val, "%u", &num_iter_per_batch ))   { ::usage( "failed to translate a value of <num_iter_per_batch>" );    return 1; }
            if( num_iter_per_batch == 0 )                        { ::usage( "<num_iter_per_batch> can't have a value of 0" );           return 1; }

        } else if( !strcmp( opt, "-p" )) {
            if( !numArgs )                                       { ::usage( "missing value for option <max_threads>" );                 return 1; }
            const char* val = *(argsPtr++); --numArgs;
            if( 1 != sscanf( val, "%lu", &numThreads ))          { ::usage( "failed to translate a value of <max_threads>" );           return 1; }
            if( numThreads == 0 )                                { ::usage( "<max_threads> can't have a value of 0" );                  return 1; }

        } else if( !strcmp( opt, "-o" )) {
            if( !numArgs )                                       { ::usage( "record value isn't following the option" );                return 1; }
            outfilename = *(argsPtr++); --numArgs;

        } else if( !strcmp( opt, "-n" )) {
            no_compression = true;

        } else if( !strcmp( opt, "-s" )) {
            stats = true;

        } else if( !strcmp( opt, "-t" )) {
            test = true;

        } else if( !strcmp( opt, "-d" )) {
            dump = true;

        } else if( !strcmp( opt, "-v" )) {
            verbose = true;

        } else                                                 { ::usage( "unknown command option" );                          return 1; }
    }
    if( numArgs )                                              { ::usage( "illegal number of parameters" );                    return 1; }

    printf( "image size (16-bit numbers): %lu\n", reclen );
    printf( "images per batch:            %lu\n", num_images_per_batch );
    printf( "iterations per batch:        %u\n", num_iter_per_batch );
    printf( "maximum number of threads:   %lu\n", numThreads );

    ::CompressionTestMT ct( reclen, num_images_per_batch, num_iter_per_batch, numThreads );

    return ct.run( infilename, outfilename, stats, test, dump, verbose, no_compression ) ? 0 : 1;
}
