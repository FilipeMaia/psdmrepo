#include "psana_test/adler.h"
#include "zlib.h"

uint64_t psana_test_adler32(const void *buffer, uint64_t n) {
  uLong adler = adler32(0L, Z_NULL, 0);
  adler = adler32(adler, (const Bytef *)buffer, n);
  return adler;
}
