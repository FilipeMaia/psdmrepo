#ifndef PSANA_TEST_ADLER_H
#define PSANA_TEST_ADLER_H

#include <stdint.h>

extern "C" {
uint64_t psana_test_adler32(const void *buffer, uint64_t n);
};

#endif
