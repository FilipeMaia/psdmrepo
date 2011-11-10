#ifndef PDS_CODEC_COMPRESSORMT_H
#define PDS_CODEC_COMPRESSORMT_H

//--------------------------------------------------------------------------
// File and Version Information:
//   $Id: $
//
// Description:
//   Class CompressorMT. A simple wrapper allowing multithreaded compression
//   for API conforming algorithms.
//
// Author:
//   Igor A. Gaponenko, SLAC National Accelerator Laboratory
//--------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include <stdio.h>
#include <pthread.h>
#include <assert.h>

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

        template< class COMPRESSOR >
        class CompressorMT {

        public:

            enum {

                Success = 0,

                // Status values returned due to incorrect input to operations
                //
                ErrNoImages,           // no images to process

                // Status values returned by algorithms
                //
                ErrStartThreads,       // failed to start threads
                ErrWrongThreadStatus,  // threads didn't finish processing the previous request (implementation bug)
                ErrInAlgorithm         // failed to process at least least one of the images in a batch
            };

            explicit CompressorMT(size_t numThreads=1);

            virtual ~CompressorMT();

            int compress(const void** image, const typename COMPRESSOR::ImageParams* params,
                         void** outData, size_t* outDataSize,
                         int* stat,
                         size_t numImages);

            int decompress(const void** outData, const size_t* outDataSize,
                           void** image, typename COMPRESSOR::ImageParams* params,
                           int* stat,
                           size_t numImages);

            static const char* err2str(int code);

        private:

            void delete_thread_data();

            void allocate_thread_data();

            int implement_request(bool compress_vs_decompress,
                                  void** image, typename COMPRESSOR::ImageParams* params,
                                  void** outData, size_t* outDataSize,
                                  int* stat,
                                  size_t numImages);

            bool start_threads();

            static void* processor(void* arg);

        private:

            /* Compressor objects left from previous compression attempts (if any).
             * If a subsequent request has an equal or lesser number of images
             * then the previously created objects will be reused. Otherwise
             * they'll be recreated.
             */
            size_t m_numImages;
  
            COMPRESSOR** m_compressor;

            /* Data structures to communicate parameters and results
             * with threads.
             */
            size_t m_maxImagesPerThread;
            size_t m_numThreads;
            size_t m_numActiveThreads;
            size_t m_threadsStarted;

            pthread_mutex_t m_guard_mutex;
            pthread_cond_t  m_data_ready_cv;
            pthread_cond_t  m_finished_processing_cv;

            struct ThreadData {

                explicit ThreadData(size_t max_images2process) :

                    num_images(0),
                    image_idx (new size_t[max_images2process]),

                    operation (RESERVED),

                    compressor(new COMPRESSOR*[max_images2process]),

                    image  (new void*[max_images2process]),
                    params (new typename COMPRESSOR::ImageParams[max_images2process]),

                    outData    (new void* [max_images2process]),
                    outDataSize(new size_t[max_images2process]),

                    stat(new int[max_images2process]),

                    hasWork2do      (false),
                    numActiveThreads(0),

                    guard_mutex           (0),
                    data_ready_cv         (0),
                    finished_processing_cv(0)

                {}

                ~ThreadData()
                {
                    delete [] image_idx;   image_idx   = 0;
                    delete [] compressor;  compressor  = 0;
                    delete [] image;       image       = 0;
                    delete [] params;      params      = 0;
                    delete [] outData;     outData     = 0;
                    delete [] outDataSize; outDataSize = 0;
                    delete [] stat;        stat        = 0;
                }

                pthread_t id;

                size_t  num_images;
                size_t* image_idx;

                enum Operation { COMPRESS, DECOMPRESS, RESERVED };
                Operation operation;

                COMPRESSOR** compressor;

                void**   image;
                typename COMPRESSOR::ImageParams* params;

                void**   outData;
                size_t*  outDataSize;

                int*     stat;

                bool     hasWork2do;
                size_t*  numActiveThreads;

                pthread_mutex_t* guard_mutex;
                pthread_cond_t*  data_ready_cv;
                pthread_cond_t*  finished_processing_cv;

            private:
                ThreadData();
                ThreadData& operator=(const ThreadData&);
            };
            ThreadData** m_thread_data;
        };
    }
}

template< class COMPRESSOR >
Pds::Codec::CompressorMT<COMPRESSOR>::CompressorMT(size_t numThreads) :

    m_numImages  (0),
    m_compressor (0),

    m_maxImagesPerThread(1),
    m_numThreads        (numThreads > 0 ? numThreads : 1),
    m_numActiveThreads  (0),
    m_threadsStarted    (false),

    m_thread_data(0)
{
    pthread_mutex_init( &m_guard_mutex, NULL);
    pthread_cond_init ( &m_data_ready_cv, NULL);
    pthread_cond_init ( &m_finished_processing_cv, NULL);

    this->allocate_thread_data();
}

template< class COMPRESSOR >
Pds::Codec::CompressorMT<COMPRESSOR>::~CompressorMT()
{
    if(m_threadsStarted) {
        for( size_t i = 0; i < m_numThreads; ++i )
            pthread_cancel( m_thread_data[i]->id );
        pthread_mutex_destroy( &m_guard_mutex );
        pthread_cond_destroy ( &m_data_ready_cv );
        pthread_cond_destroy ( &m_finished_processing_cv );
    }
    if( 0 != m_compressor ) {
        for( size_t i = 0; i < m_numImages; ++i ) delete m_compressor[i];
        delete [] m_compressor;
        m_compressor = 0;
    }
    this->delete_thread_data();
}

template< class COMPRESSOR >
int
Pds::Codec::CompressorMT<COMPRESSOR>::compress(
    const void** image, const typename COMPRESSOR::ImageParams* params,
    void** outData, size_t* outDataSize,
    int* stat,
    size_t numImages)
{
    return this->implement_request(
        true,
        (void**)image, (typename COMPRESSOR::ImageParams*)params,
        outData, outDataSize,
        stat,
        numImages);
}

template< class COMPRESSOR >
int
Pds::Codec::CompressorMT<COMPRESSOR>::decompress(
    const void** outData, const size_t* outDataSize,
    void** image, typename COMPRESSOR::ImageParams* params,
    int* stat,
    size_t numImages)
{
    return this->implement_request(
        false,
        image, params,
        (void**)outData, (size_t*)outDataSize,
        stat,
        numImages);
}


template< class COMPRESSOR >
void
Pds::Codec::CompressorMT<COMPRESSOR>::delete_thread_data()
{
    for( size_t i = 0; i < m_numThreads; ++i ) delete m_thread_data[i];
    delete [] m_thread_data;
    m_thread_data = 0;
}


template< class COMPRESSOR >
void
Pds::Codec::CompressorMT<COMPRESSOR>::allocate_thread_data()
{
    m_thread_data = new ThreadData*[m_numThreads];
    for( size_t i = 0; i < m_numThreads; ++i ) {
        m_thread_data[i] = new ThreadData( m_maxImagesPerThread );
        m_thread_data[i]->numActiveThreads       = &m_numActiveThreads;
        m_thread_data[i]->guard_mutex            = &m_guard_mutex;
        m_thread_data[i]->data_ready_cv          = &m_data_ready_cv;
        m_thread_data[i]->finished_processing_cv = &m_finished_processing_cv;
    }
}

template< class COMPRESSOR >
int
Pds::Codec::CompressorMT<COMPRESSOR>::implement_request(
    bool compress_vs_decompress,
    void** image, typename COMPRESSOR::ImageParams* params,
    void** outData, size_t* outDataSize,
    int* stat,
    size_t numImages)
{
    if( 0 == numImages ) return ErrNoImages;  // not enough images to process

    // Calculate the number of images per thread based on the specified
    // (in the c-tor) number of threads and the number of images.
    //
    // Expand the thread data to accommodate more images per thread if needed.
    //
    size_t maxImagesPerThread = numImages < m_numThreads ? m_numThreads : (numImages + 1) / m_numThreads;

    if( maxImagesPerThread > m_maxImagesPerThread ) {
        this->delete_thread_data();
        m_maxImagesPerThread = maxImagesPerThread;
        this->allocate_thread_data();
    }

    // Expand the list of compressors if needed
    //
    if( numImages > m_numImages ) {
        for( size_t i = 0; i < m_numImages; ++i ) delete m_compressor[i];
        delete [] m_compressor;
        m_numImages = numImages;
        m_compressor = new COMPRESSOR*[m_numImages];
        for( size_t i = 0; i < m_numImages; ++i ) m_compressor[i] = new COMPRESSOR();
    }

    if( !this->start_threads()) return ErrStartThreads;

    size_t image2process = 0;

    while( true ) {

        pthread_mutex_lock( &m_guard_mutex );

        // Make sure no active threads are left from a previous invocation
        // of the operation.
        //
        if( 0 != m_numActiveThreads ) {
            pthread_mutex_unlock( &m_guard_mutex );
            assert(0);
            return ErrWrongThreadStatus;
        }

        for( size_t thread_idx = 0; thread_idx < m_numThreads; ++thread_idx ) {

            size_t numImagesPerThread = 0;

            // ATTENTION: This loop uses the actual value of the parameter calculate
            //            for the current request, not the maximum capacity of the thread
            //            processors, which might be higher in previous requests. Otherwise
            //            images won't be spread equally between threads.
            //
            for( size_t i = 0; i < maxImagesPerThread; ++i ) {

                if( image2process >= numImages ) break;  // no more images are left to process

                m_thread_data[thread_idx]->image_idx [i] = image2process;
                m_thread_data[thread_idx]->operation     = compress_vs_decompress ? ThreadData::COMPRESS : ThreadData::DECOMPRESS;
                m_thread_data[thread_idx]->compressor[i] = m_compressor[image2process];
                if( compress_vs_decompress ) {
                    m_thread_data[thread_idx]->image      [i] = image [image2process];
                    m_thread_data[thread_idx]->params     [i] = params[image2process];
                    m_thread_data[thread_idx]->outData    [i] = 0;
                    m_thread_data[thread_idx]->outDataSize[i] = 0;
                } else {
                    m_thread_data[thread_idx]->image      [i] = 0;
                    m_thread_data[thread_idx]->outData    [i] = outData    [image2process];
                    m_thread_data[thread_idx]->outDataSize[i] = outDataSize[image2process];
                }
                m_thread_data[thread_idx]->stat[i] = 0;

                numImagesPerThread++;
                image2process++;
            }
            if( 0 == numImagesPerThread ) break;  // no more images are left to process

            m_thread_data[thread_idx]->num_images = numImagesPerThread;
            m_thread_data[thread_idx]->hasWork2do = true;

            m_numActiveThreads++;
        }
        if( 0 == m_numActiveThreads ) {  // no more images are left to process
            pthread_mutex_unlock( &m_guard_mutex );
            break;
        }

        const size_t numThreadsStarted = m_numActiveThreads;  // store this value because threads
                                                              // are going to decrement it when finishing
                                                              // processing their assignments.

        pthread_cond_broadcast( &m_data_ready_cv );
        pthread_mutex_unlock( &m_guard_mutex );

        pthread_mutex_lock( &m_guard_mutex );
        while( m_numActiveThreads )
            pthread_cond_wait( &m_finished_processing_cv, &m_guard_mutex );

        for( size_t thread_idx = 0; thread_idx < numThreadsStarted; ++thread_idx ) {
            for( size_t i = 0; i < m_thread_data[thread_idx]->num_images; ++i ) {
                const size_t image_idx = m_thread_data[thread_idx]->image_idx[i];
                if( compress_vs_decompress ) {
                    outData    [image_idx] = m_thread_data[thread_idx]->outData    [i];
                    outDataSize[image_idx] = m_thread_data[thread_idx]->outDataSize[i];
                } else {
                    image [image_idx] = m_thread_data[thread_idx]->image [i];
                    params[image_idx] = m_thread_data[thread_idx]->params[i];
                }
                stat[image_idx] = m_thread_data[thread_idx]->stat[i];
            }
        }
        pthread_mutex_unlock( &m_guard_mutex );
    }

    // Harvest status values returned by each compressor and return a summary
    // flag indicating success if all images have been successfully processed.
    //
    for( size_t i = 0; i < numImages; ++i )
        if( 0 != stat[i] )
            return ErrInAlgorithm;  // at least one of the images hasn't been successfully processed.
                                    // see details in the returned status array.

    return Success;
}

template< class COMPRESSOR >
bool
Pds::Codec::CompressorMT<COMPRESSOR>::start_threads()
{
    if( !m_threadsStarted ) {
        for( size_t i = 0; i < m_numThreads; ++i )
            if(pthread_create(&(m_thread_data[i]->id), NULL, processor, (void*)(m_thread_data[i])))
                return false;
        m_threadsStarted = true;
    }
    return true;
}

template< class COMPRESSOR >
void*
Pds::Codec::CompressorMT<COMPRESSOR>::processor(void* arg)
{
    ThreadData* td = (ThreadData*)arg;

    //printf("DEBUG: thread %u process image %u ptr=%llu\n", td->id, td->image_idx, td->image );

    while( true ) {

        // Wait for a processing assignment
        //
        pthread_mutex_lock( td->guard_mutex );

        while( !td->hasWork2do )
            pthread_cond_wait( td->data_ready_cv, td->guard_mutex );

        td->hasWork2do = false;

        pthread_mutex_unlock( td->guard_mutex );

        // Analyze and process the request
        //
        switch( td->operation ) {

        case ThreadData::COMPRESS:
            for( size_t i = 0; i < td->num_images; ++i )
                td->stat[i] = td->compressor[i]->compress(
                    td->image[i],
                    td->params[i],
                    td->outData[i],
                    td->outDataSize[i] );
            break;

        case ThreadData::DECOMPRESS:
            for( size_t i = 0; i < td->num_images; ++i )
                td->stat[i] = td->compressor[i]->decompress(
                    td->outData[i],
                    td->outDataSize[i],
                    td->image[i],
                    td->params[i] );
            break;

        default:
            assert(0);
        }

        // Notify the main thread that the job is done in case if this is the last
        // active thread.
        //
        pthread_mutex_lock( td->guard_mutex );

        if( 0 == *(td->numActiveThreads)) assert(0);  // algorithm implementation error

        if( 0 == --(*(td->numActiveThreads)))
            pthread_cond_signal( td->finished_processing_cv );

        pthread_mutex_unlock( td->guard_mutex );

    }
    //printf("DEBUG: thread: %u\n", td->id );
    return NULL;
}

template< class COMPRESSOR >
const char*
Pds::Codec::CompressorMT<COMPRESSOR>::err2str(int code)
{
    switch( code ) {
    case Success:              return "Success";
    case ErrNoImages:          return "No images to process";
    case ErrStartThreads:      return "Failed to start threads";
    case ErrWrongThreadStatus: return "Threads didn't finish processing the previous request (implementation bug)";
    case ErrInAlgorithm:       return "Failed to process at least least one of the images in a batch";
    }
}

#endif  // PDS_CODEC_COMPRESSORMT_H
