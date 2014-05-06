#ifndef PSANATEST_DUMPBASICTYPES_H
#define PSANATEST_DUMPBASICTYPES_H

#include <string>
#include <stdint.h>
#include "ndarray/ndarray.h"

namespace psana_test {

std::string dump_int8(int8_t v);
std::string dump_int16(int16_t v);
std::string dump_int32(int32_t v);
std::string dump_int64(int64_t v);
std::string dump_uint8(uint8_t v);
std::string dump_uint16(uint16_t v);
std::string dump_uint32(uint32_t v);
std::string dump_uint64(uint64_t v);
std::string dump_str(const char *v);
std::string dump_float(float v);
std::string dump_double(double v);

std::string dump_ndarray_float32_1(const ndarray<const float,1> &a);
std::string dump_ndarray_float32_2(const ndarray<const float,2> &a);
std::string dump_ndarray_float32_3(const ndarray<const float,3> &a);
std::string dump_ndarray_float32_4(const ndarray<const float,4> &a);
std::string dump_ndarray_float32_5(const ndarray<const float,5> &a);
std::string dump_ndarray_float32_6(const ndarray<const float,6> &a);
std::string dump_ndarray_float64_1(const ndarray<const double,1> &a);
std::string dump_ndarray_float64_2(const ndarray<const double,2> &a);
std::string dump_ndarray_float64_3(const ndarray<const double,3> &a);
std::string dump_ndarray_float64_4(const ndarray<const double,4> &a);
std::string dump_ndarray_float64_5(const ndarray<const double,5> &a);
std::string dump_ndarray_float64_6(const ndarray<const double,6> &a);
std::string dump_ndarray_int16_1(const ndarray<const int16_t,1> &a);
std::string dump_ndarray_int16_2(const ndarray<const int16_t,2> &a);
std::string dump_ndarray_int16_3(const ndarray<const int16_t,3> &a);
std::string dump_ndarray_int16_4(const ndarray<const int16_t,4> &a);
std::string dump_ndarray_int16_5(const ndarray<const int16_t,5> &a);
std::string dump_ndarray_int16_6(const ndarray<const int16_t,6> &a);
std::string dump_ndarray_int32_1(const ndarray<const int32_t,1> &a);
std::string dump_ndarray_int32_2(const ndarray<const int32_t,2> &a);
std::string dump_ndarray_int32_3(const ndarray<const int32_t,3> &a);
std::string dump_ndarray_int32_4(const ndarray<const int32_t,4> &a);
std::string dump_ndarray_int32_5(const ndarray<const int32_t,5> &a);
std::string dump_ndarray_int32_6(const ndarray<const int32_t,6> &a);
std::string dump_ndarray_int64_1(const ndarray<const int64_t,1> &a);
std::string dump_ndarray_int64_2(const ndarray<const int64_t,2> &a);
std::string dump_ndarray_int64_3(const ndarray<const int64_t,3> &a);
std::string dump_ndarray_int64_4(const ndarray<const int64_t,4> &a);
std::string dump_ndarray_int64_5(const ndarray<const int64_t,5> &a);
std::string dump_ndarray_int64_6(const ndarray<const int64_t,6> &a);
std::string dump_ndarray_uint8_1(const ndarray<const uint8_t,1> &a);
std::string dump_ndarray_uint8_2(const ndarray<const uint8_t,2> &a);
std::string dump_ndarray_uint8_3(const ndarray<const uint8_t,3> &a);
std::string dump_ndarray_uint8_4(const ndarray<const uint8_t,4> &a);
std::string dump_ndarray_uint8_5(const ndarray<const uint8_t,5> &a);
std::string dump_ndarray_uint8_6(const ndarray<const uint8_t,6> &a);

 } // namespace psana_test

#endif
