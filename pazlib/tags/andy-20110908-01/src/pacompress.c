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

#define PAZLIB_DEBUG 0

#if PAZLIB_DEBUG
#define DBGMSG(x) fprintf x;
#else
#define DBGMSG(x)
#endif


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
  if (flush == Z_FINISH) {
    if (err != Z_STREAM_END) {
        deflateEnd(&stream);
        return err == Z_OK ? Z_BUF_ERROR : err;
    }
  } else {
    if (err != Z_OK) {
        deflateEnd(&stream);
        return err;
    }
  }
  *destLen = stream.total_out;

  if (adler) *adler = stream.adler;

  err = deflateEnd(&stream);
  if (flush != Z_FINISH && err == Z_DATA_ERROR) err = Z_OK;

  return err;
}

static int get_cpu_count()
{
  static int cpuCount = 0;
  if (cpuCount == 0) {
    char* env = getenv("PAZLIB_MAX_THREADS");
    if (env) {
      cpuCount = atoi(env);
    }
  }
  if (cpuCount == 0) {
    cpuCount = sysconf(_SC_NPROCESSORS_ONLN);
  }

  return cpuCount;
}


typedef struct thread_data {

  const Bytef *source;
  uLong sourceLen;
  Bytef *dest;
  uLongf destLen;
  int level;
  int err;
  uLong adler;

} thread_data;


static void* comp_thread(void* th_data)
{
  thread_data* data = (thread_data*)th_data;

  data->err = _compress(data->dest, &data->destLen,
      data->source, data->sourceLen, data->level, Z_FULL_FLUSH, &data->adler);

  return th_data;
}


/*
//      ----------------------------------------
//      -- Public Function Member Definitions --
//      ----------------------------------------
*/

int ZEXPORT compress2(Bytef *dest, uLongf *destLen,
    const Bytef *source, uLong sourceLen, int level)
{
  int numThreads;
  int thread;
  uLong bytesPerThread;
  uLong outBytesPerThread;
  thread_data* th_data = NULL;
  pthread_t* threads = NULL;
  uLong total_data_size = 0;
  int err = Z_OK;
  uLong adler = 0;

  /*
   * Guess number of threads we could potentially use for this particular input.
   * Require that each thread receives minimum 64k bytes of data
   */
  numThreads = sourceLen / 0x10000 - 1;
  DBGMSG((stderr, "pazlib: numThreads=%d\n", numThreads))

  if (numThreads >= 2) {
    int cpu_count = get_cpu_count();
    if (numThreads > cpu_count) {
      /* Limit the number of threads by actual CPU count */
      numThreads = cpu_count;
      DBGMSG((stderr, "pazlib: cpuCount=%d\n", cpu_count))
    }
  }

  if (numThreads < 2) {
    /* do not run threads, compress it all here */
    DBGMSG((stderr, "pazlib: using standard compress\n"))
    int err = _compress(dest, destLen, source, sourceLen, level, Z_FINISH, NULL);
    DBGMSG((stderr, "pazlib: sourceLen=%lu destLen=%lu err=%d\n", sourceLen, *destLen, err))
    return err;
  }

  /* number of bytes per thread */
  bytesPerThread = sourceLen / numThreads;
  outBytesPerThread = bytesPerThread + bytesPerThread/100 + 12;
  DBGMSG((stderr, "pazlib: bytes per thread = %lu\n", bytesPerThread))

  /* fill thread data */
  th_data = (thread_data*)calloc(numThreads, sizeof(thread_data));
  if (! th_data) {
    err = Z_MEM_ERROR;
    goto done;
  }
  for (thread = 0; thread < numThreads; ++ thread) {

    th_data[thread].source = source + thread*bytesPerThread;
    if (thread+1 == numThreads) {
      th_data[thread].sourceLen = sourceLen - thread*bytesPerThread;
    } else {
      th_data[thread].sourceLen = bytesPerThread;
    }
    th_data[thread].level = level;

    // allocate per-thread output buffer
    th_data[thread].dest = malloc(outBytesPerThread);
    if (! th_data[thread].dest) {
      err = Z_MEM_ERROR;
      goto done;
    }
    th_data[thread].destLen = outBytesPerThread;

    DBGMSG((stderr, "pazlib: th_data[%d]: sourceLen=%lu destLen=%lu\n", thread, th_data[thread].sourceLen, th_data[thread].destLen))
  }

  /* start threads */
  threads = (pthread_t*)malloc(sizeof(pthread_t)*numThreads);
  if (! threads) {
    err = Z_MEM_ERROR;
    goto done;
  }
  for (thread = 0; thread < numThreads; ++ thread) {
    int s = pthread_create(&threads[thread], NULL, comp_thread, &th_data[thread]);
    DBGMSG((stderr, "pazlib: thread %d started, code: %d\n", thread, s))
  }

  /* wait all threads to finish */
  for (thread = 0; thread < numThreads; ++ thread) {
    int s = pthread_join(threads[thread], NULL);
    DBGMSG((stderr, "pazlib: thread %d joined, code: %d\n", thread, s))

    DBGMSG((stderr, "pazlib: th_data[%d]: destLen=%lu err=%d\n", thread, th_data[thread].destLen, th_data[thread].err))
  }


  /* calculate total size of the data */
  for (thread = 0; thread < numThreads; ++ thread) {
    /* exclude 2-byte header */
    total_data_size += th_data[thread].destLen - 2;
  }
  /* add 2-byte header, 5-byte "last" buffer, and 4-byte checksum */
  total_data_size += 2 + 5 + 4;

  if (total_data_size > *destLen) {
    err = Z_BUF_ERROR;
    goto done;
  }

  /* copy the data */
  for (thread = 0; thread < numThreads; ++ thread) {
    Bytef* src = th_data[thread].dest;
    uLong len = th_data[thread].destLen;
    if (thread != 0) {
      /* skip header except for the first one */
      src += 2;
      len -= 2;
    }
    memcpy(dest, src, len);

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

done:

  /* free everything that has been allocated */
  if (th_data) {
    for (thread = 0; thread < numThreads; ++ thread) {
      free(th_data[thread].dest);
    }
    free(th_data);
  }
  free(threads);

  return err;

}


