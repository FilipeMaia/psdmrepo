/*
//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class pacompress...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------
*/
/*
//-----------------------
// This Class's Header --
//-----------------------
*/

/*
//-----------------
// C/C++ Headers --
//-----------------
 */
#include <string.h>
#include <stdio.h>
#include <zlib.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>

/*
//-------------------------------
// Collaborating Class Headers --
//-------------------------------
*/

/*
//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
*/

#define PAZLIB_MAX_THREADS 64

#define PAZLIB_DEBUG 0

#if PAZLIB_DEBUG
#define DBGMSG(x) fprintf x;
#else
#define DBGMSG(x)
#endif

/*
 * data passed to a thread, includes some sync stuff
 */
typedef struct thread_data {

  int index;
  
  const Bytef *source;  /* data to compress */
  uLong sourceLen;

  Bytef *dest;      /* compressed data */
  uLongf destLen;
  
  int level;   /* compression level */
  int err;     /* return code from zlib */
  uLong adler; /* CRC for this block */
  
  pthread_mutex_t src_mutex;
  pthread_cond_t src_cond;

  pthread_mutex_t dst_mutex;
  pthread_cond_t dst_cond;

} thread_data;

pthread_mutex_t pazlib_sync;     /* global mutex to serialize all calls to pazlib */
static int threads_running = 0;  /* number of threads created so far */
static pthread_t threads[PAZLIB_MAX_THREADS];    /* array for threads */
static thread_data th_data[PAZLIB_MAX_THREADS];  /* array with thread data */

static void* comp_thread(void*);  /* function wich runs in thread */

/* constructor for global mutex */
void pazlib_init(void) __attribute__((constructor));
void pazlib_init(void)
{
  DBGMSG((stderr, "pazlib_init: starting initialization\n"))
  int s = pthread_mutex_init(&pazlib_sync, NULL);
  if (s != 0) {
    fprintf(stderr, "pazlib_init: pthread_mutex_init failed: %s\n", strerror(s));
    abort();
  }
}

/* start one thread */
static int start_new_thread(int index) 
{ 
  int s;
  
  /* init mutextes/cond vars */
  s = pthread_mutex_init(&th_data[index].src_mutex, NULL);
  if (s != 0) return s;
  s = pthread_mutex_init(&th_data[index].dst_mutex, NULL);
  if (s != 0) return s;
  s = pthread_cond_init(&th_data[index].src_cond, NULL);
  if (s != 0) return s;
  s = pthread_cond_init(&th_data[index].dst_cond, NULL);
  if (s != 0) return s;
  
  /* set no-data flag */
  th_data[index].source = NULL;

  th_data[index].index = index;

  /* start thread */
  s = pthread_create(&threads[index], NULL, comp_thread, &th_data[index]);
  DBGMSG((stderr, "pazlib: thread %d started, code: %d\n", index, s))
  return s;
}

/* start additional threads so that total number of threads is
 * at least num_threads
 */
static int start_threads(int num_threads)
{
  for (; threads_running < num_threads; ++ threads_running) {
    int s = start_new_thread(threads_running);
    if (s != 0) return s;
  }
  return 0;
}

/* compression routine, modified version of zlib compress2() */
static int _compress(Bytef *dest, uLongf *destLen,
    const Bytef *source, uLong sourceLen,
    int level, int flush, uLong* adler)
{
  z_stream stream;
  int err;

  stream.next_in = (Bytef*)source;
  stream.avail_in = (uInt)sourceLen;

  stream.next_out = dest;
  stream.avail_out = (uInt)*destLen;
  if ((uLong)stream.avail_out != *destLen) return Z_BUF_ERROR;

  stream.zalloc = (alloc_func)0;
  stream.zfree = (free_func)0;
  stream.opaque = (voidpf)0;

  err = deflateInit(&stream, level);
  if (err != Z_OK) return err;

  err = deflate(&stream, flush);
  DBGMSG((stderr, "pazlib: deflate: avail_in=%u, total_in=%lu, avail_out=%u, total_out=%lu, err=%d\n",
      stream.avail_in, stream.total_in, stream.avail_out, stream.total_out, err))
  /* check that complete input buffer was compressed and result fits in output buffer*/
  if (flush == Z_FINISH) {
    if (err != Z_STREAM_END) {
        deflateEnd(&stream);
        return err == Z_OK ? Z_BUF_ERROR : err;
    }
  } else {
    if (err != Z_OK) {
        deflateEnd(&stream);
        return err;
    } else  if (stream.avail_out == 0) {
      deflateEnd(&stream);
      return Z_BUF_ERROR;
    }
  }
  *destLen = stream.total_out;

  if (adler) *adler = stream.adler;

  err = deflateEnd(&stream);
  if (flush != Z_FINISH && err == Z_DATA_ERROR) err = Z_OK;

  return err;
}

/* get maximum thread count */
static int get_thread_count()
{
  static int threadCount = 0;
  if (threadCount == 0) {
    char* env = getenv("PAZLIB_MAX_THREADS");
    if (env) {
        threadCount = atoi(env);
    }
    if (threadCount == 0) {
        threadCount = sysconf(_SC_NPROCESSORS_ONLN);
    }
    if (threadCount > PAZLIB_MAX_THREADS) threadCount = PAZLIB_MAX_THREADS;
  }
  
  return threadCount;
}

/* thread function which waits for data on a condition,
 * compresses the data and signals back after compression is finished.
 */
static void* comp_thread(void* th_data)
{
  int rc;
  thread_data* data = (thread_data*)th_data;

  while (1) {
  
    /* wait for new data */
    DBGMSG((stderr, "pazlib: thread %d locking src_mutex\n", data->index))
    rc = pthread_mutex_lock(&data->src_mutex);
    if (rc) {
      fprintf(stderr, "pazlib: pthread_mutex_lock failed: %s\n", strerror(rc));
      abort();
    }
    while(! data->source) {
      DBGMSG((stderr, "pazlib: thread %d wait on src_cond\n", data->index))
      rc = pthread_cond_wait(&data->src_cond, &data->src_mutex);
      if (rc) {
        fprintf(stderr, "pazlib: pthread_cond_wait failed: %s\n", strerror(rc));
        abort();
      }
    }
    DBGMSG((stderr, "pazlib: thread %d unlocking src_mutex\n", data->index))
    rc = pthread_mutex_unlock(&data->src_mutex);
    if (rc) {
      fprintf(stderr, "pazlib: pthread_mutex_unlock failed: %s\n", strerror(rc));
      abort();
    }
    
    DBGMSG((stderr, "pazlib: thread %d locking dst_mutex\n", data->index))
    rc = pthread_mutex_lock(&data->dst_mutex);
    if (rc) {
      fprintf(stderr, "pazlib: pthread_mutex_lock failed: %s\n", strerror(rc));
      abort();
    }

    data->err = _compress(data->dest, &data->destLen,
        data->source, data->sourceLen, data->level, Z_FULL_FLUSH, &data->adler);

    /* reset source so that next iteration stops */
    data->source = NULL;
    
    /* notify caller that we got compressed data */
    DBGMSG((stderr, "pazlib: thread %d unlocking dst_mutex\n", data->index))
    rc = pthread_mutex_unlock(&data->dst_mutex);
    if (rc) {
      fprintf(stderr, "pazlib: pthread_cond_signal failed: %s\n", strerror(rc));
      abort();
    }
    DBGMSG((stderr, "pazlib: thread %d signal dst_cond\n", data->index))
    rc = pthread_cond_signal(&data->dst_cond);
    if (rc) {
      fprintf(stderr, "pazlib: pthread_cond_signal failed: %s\n", strerror(rc));
      abort();
    }

  }
  
  return th_data;
}


/*
//      ----------------------------------------
//      -- Public Function Member Definitions --
//      ----------------------------------------
*/

#define RETURN(x) do {rc=x; goto done;} while(1);

int ZEXPORT compress2(Bytef *dest, uLongf *destLen,
    const Bytef *source, uLong sourceLen, int level)
{
  int numThreads;
  int thread;
  int rc;
  uLong bytesPerThread;
  uLong outBytesPerThread;
  uLong total_data_size = 0;
  uLong adler = 0;

  /*
   * Guess number of threads we could potentially use for this particular input.
   * Require that each thread receives minimum 64k bytes of data
   */
  numThreads = sourceLen / 0x10000 - 1;
  DBGMSG((stderr, "pazlib: numThreads=%d\n", numThreads))

  /* limit it by maximum thread count */
  if (numThreads >= 2) {
    int max_thread_count = get_thread_count();
    if (numThreads > max_thread_count) {
      /* Limit the number of threads by actual CPU count */
      numThreads = max_thread_count;
      DBGMSG((stderr, "pazlib: cpuCount=%d\n", max_thread_count))
    }
  }

  if (numThreads < 2) {
    /* do not run threads, compress it all here */
    DBGMSG((stderr, "pazlib: using standard compress\n"))
    int err = _compress(dest, destLen, source, sourceLen, level, Z_FINISH, NULL);
    DBGMSG((stderr, "pazlib: sourceLen=%lu destLen=%lu err=%d\n", sourceLen, *destLen, err))
    return err;
  }

  /* we are not thread-safe, so lock global mutex to serialize everything after here */
  DBGMSG((stderr, "pazlib: lock global mutex\n", thread))
  rc = pthread_mutex_lock(&pazlib_sync);
  if (rc) {
    fprintf(stderr, "pazlib: pthread_mutex_lock failed: %s\n", strerror(rc));
    RETURN(Z_MEM_ERROR);
  }

  /* start more threads if needed */
  rc = start_threads(numThreads);
  if (rc != 0) {
    fprintf(stderr, "pazlib: failed to start threads: %s\n", strerror(rc));
    RETURN(Z_MEM_ERROR);
  }
  
  /* number of bytes per thread in input and outbut buffers */
  bytesPerThread = sourceLen / numThreads;
  outBytesPerThread = *destLen / numThreads;
  DBGMSG((stderr, "pazlib: bytes per thread src = %lu dst = %lu\n", bytesPerThread, outBytesPerThread))

  /* pass the data to all threads */
  for (thread = 0; thread < numThreads; ++ thread) {

    DBGMSG((stderr, "pazlib: lock src_mutex[%d]\n", thread))
    rc = pthread_mutex_lock(&th_data[thread].src_mutex);
    if (rc) {
      fprintf(stderr, "pazlib: pthread_mutex_lock failed: %s\n", strerror(rc));
      RETURN(Z_MEM_ERROR);
    }

    /* piece of input data */
    th_data[thread].source = source + thread*bytesPerThread;
    if (thread+1 == numThreads) {
      th_data[thread].sourceLen = sourceLen - thread*bytesPerThread;
    } else {
      th_data[thread].sourceLen = bytesPerThread;
    }

    /* per-thread output buffer */
    th_data[thread].dest = dest + thread*outBytesPerThread;
    th_data[thread].destLen = outBytesPerThread;

    th_data[thread].level = level;

    /* set err to special value to watch it changes */
    th_data[thread].err = -999;

    DBGMSG((stderr, "pazlib: unlock src_mutex[%d]\n", thread))
    rc = pthread_mutex_unlock(&th_data[thread].src_mutex);
    if (rc) {
      fprintf(stderr, "pazlib: pthread_mutex_unlock failed: %s\n", strerror(rc));
      RETURN(Z_MEM_ERROR);
    }

    /* notify thread */
    DBGMSG((stderr, "pazlib: signal src_cond[%d]\n", thread))
    rc = pthread_cond_signal(&th_data[thread].src_cond);
    if (rc) {
      fprintf(stderr, "pazlib: pthread_cond_signal failed: %s\n", strerror(rc));
      RETURN(Z_MEM_ERROR);
    }
    
    DBGMSG((stderr, "pazlib: th_data[%d]: sourceLen=%lu destLen=%lu\n", thread, th_data[thread].sourceLen, th_data[thread].destLen))
  }

  /* wait all threads to finish */
  for (thread = 0; thread < numThreads; ++ thread) {

    /* wait for compressed data */
    DBGMSG((stderr, "pazlib: lock dst_mutex[%d]\n", thread))
    rc = pthread_mutex_lock(&th_data[thread].dst_mutex);
    if (rc) {
      fprintf(stderr, "pazlib: pthread_mutex_lock failed: %s\n", strerror(rc));
      RETURN(Z_MEM_ERROR);
    }
    while(th_data[thread].err == -999) {
      DBGMSG((stderr, "pazlib: wait on dst_cond[%d]\n", thread))
      rc = pthread_cond_wait(&th_data[thread].dst_cond, &th_data[thread].dst_mutex);
      if (rc) {
        fprintf(stderr, "pazlib: pthread_cond_wait failed: %s\n", strerror(rc));
        RETURN(Z_MEM_ERROR);
      }
    }
    DBGMSG((stderr, "pazlib: unlock dst_mutex[%d]\n", thread))
    rc = pthread_mutex_unlock(&th_data[thread].dst_mutex);
    if (rc) {
      fprintf(stderr, "pazlib: pthread_mutex_unlock failed: %s\n", strerror(rc));
      RETURN(Z_MEM_ERROR);
    }
    
    DBGMSG((stderr, "pazlib: th_data[%d]: destLen=%lu err=%d\n", thread, th_data[thread].destLen, th_data[thread].err))
  }

  /* check all return codes */
  for (thread = 0; thread < numThreads; ++ thread) {
    if (th_data[thread].err != Z_OK) RETURN(th_data[thread].err);
  }

  /* calculate total size of the data */
  for (thread = 0; thread < numThreads; ++ thread) {
    /* exclude 2-byte header */
    total_data_size += th_data[thread].destLen - 2;
  }
  /* add 2-byte header, 5-byte "last" buffer, and 4-byte checksum */
  total_data_size += 2 + 5 + 4;
  if (total_data_size > *destLen) {
    RETURN(Z_BUF_ERROR);
  }

  /* copy the data */
  for (thread = 0; thread < numThreads; ++ thread) {
    Bytef* src = th_data[thread].dest;
    uLong len = th_data[thread].destLen;
    if (thread != 0) {
      
      /* skip header except for the first one */
      src += 2;
      len -= 2;
      
      /* sorce and destination may overlap */
      memmove(dest, src, len);
    }

    dest += len;
  }

  /* add final empty block */
  *dest++ = 0x1;
  *dest++ = 0x0;
  *dest++ = 0x0;
  *dest++ = 0xff;
  *dest++ = 0xff;

  /* recalculate adler32 from all fragments */
  adler = adler32(0L, NULL, 0);
  for (thread = 0; thread < numThreads; ++ thread) {
    adler = adler32_combine(adler, th_data[thread].adler, th_data[thread].sourceLen);
    DBGMSG((stderr, "pazlib: th_data[%d]: adler=%lu sum_adler=%lu\n", thread, th_data[thread].adler, adler))
  }

  /* add adler checksum */
  *dest++ = (adler >> 24) & 0xff;
  *dest++ = (adler >> 16) & 0xff;
  *dest++ = (adler >> 8) & 0xff;
  *dest++ = adler & 0xff;

  /* store correct buffer size */
  *destLen = total_data_size;

  RETURN(Z_OK);

done:

  /* unlock global mutex */
  pthread_mutex_unlock(&pazlib_sync);
  return rc;

}


