#ifndef PSANATEST_TYPES_TO_STR_H
#define PSANATEST_TYPES_TO_STR_H

#include <string>
#include <stdint.h>
#include "ndarray/ndarray.h"

namespace psana_test {

std::string int8_to_str(int8_t v, int indent=0, int lvl=0);
std::string int16_to_str(int16_t v, int indent=0, int lvl=0);
std::string int32_to_str(int32_t v, int indent=0, int lvl=0);
std::string int64_to_str(int64_t v, int indent=0, int lvl=0);
std::string uint8_to_str(uint8_t v, int indent=0, int lvl=0);
std::string uint16_to_str(uint16_t v, int indent=0, int lvl=0);
std::string uint32_to_str(uint32_t v, int indent=0, int lvl=0);
std::string uint64_to_str(uint64_t v, int indent=0, int lvl=0);

std::string str_to_str(const char *v, int indent=0, int lvl=0);
std::string fixedstr_to_str(const char *v, long n, int indent=0, int lvl=0);
std::string float_to_str(float v, int indent=0, int lvl=0);
std::string double_to_str(double v, int indent=0, int lvl=0);

std::string ndarray_float32_1_to_str(const ndarray<const float,1> &a, int indent=0, int lvl=0);
std::string ndarray_float32_2_to_str(const ndarray<const float,2> &a, int indent=0, int lvl=0);
std::string ndarray_float32_3_to_str(const ndarray<const float,3> &a, int indent=0, int lvl=0);
std::string ndarray_float32_4_to_str(const ndarray<const float,4> &a, int indent=0, int lvl=0);
std::string ndarray_float32_5_to_str(const ndarray<const float,5> &a, int indent=0, int lvl=0);
std::string ndarray_float32_6_to_str(const ndarray<const float,6> &a, int indent=0, int lvl=0);
std::string ndarray_float64_1_to_str(const ndarray<const double,1> &a, int indent=0, int lvl=0);
std::string ndarray_float64_2_to_str(const ndarray<const double,2> &a, int indent=0, int lvl=0);
std::string ndarray_float64_3_to_str(const ndarray<const double,3> &a, int indent=0, int lvl=0);
std::string ndarray_float64_4_to_str(const ndarray<const double,4> &a, int indent=0, int lvl=0);
std::string ndarray_float64_5_to_str(const ndarray<const double,5> &a, int indent=0, int lvl=0);
std::string ndarray_float64_6_to_str(const ndarray<const double,6> &a, int indent=0, int lvl=0);
std::string ndarray_int16_1_to_str(const ndarray<const int16_t,1> &a, int indent=0, int lvl=0);
std::string ndarray_int16_2_to_str(const ndarray<const int16_t,2> &a, int indent=0, int lvl=0);
std::string ndarray_int16_3_to_str(const ndarray<const int16_t,3> &a, int indent=0, int lvl=0);
std::string ndarray_int16_4_to_str(const ndarray<const int16_t,4> &a, int indent=0, int lvl=0);
std::string ndarray_int16_5_to_str(const ndarray<const int16_t,5> &a, int indent=0, int lvl=0);
std::string ndarray_int16_6_to_str(const ndarray<const int16_t,6> &a, int indent=0, int lvl=0);
std::string ndarray_int32_1_to_str(const ndarray<const int32_t,1> &a, int indent=0, int lvl=0);
std::string ndarray_int32_2_to_str(const ndarray<const int32_t,2> &a, int indent=0, int lvl=0);
std::string ndarray_int32_3_to_str(const ndarray<const int32_t,3> &a, int indent=0, int lvl=0);
std::string ndarray_int32_4_to_str(const ndarray<const int32_t,4> &a, int indent=0, int lvl=0);
std::string ndarray_int32_5_to_str(const ndarray<const int32_t,5> &a, int indent=0, int lvl=0);
std::string ndarray_int32_6_to_str(const ndarray<const int32_t,6> &a, int indent=0, int lvl=0);
std::string ndarray_int64_1_to_str(const ndarray<const int64_t,1> &a, int indent=0, int lvl=0);
std::string ndarray_int64_2_to_str(const ndarray<const int64_t,2> &a, int indent=0, int lvl=0);
std::string ndarray_int64_3_to_str(const ndarray<const int64_t,3> &a, int indent=0, int lvl=0);
std::string ndarray_int64_4_to_str(const ndarray<const int64_t,4> &a, int indent=0, int lvl=0);
std::string ndarray_int64_5_to_str(const ndarray<const int64_t,5> &a, int indent=0, int lvl=0);
std::string ndarray_int64_6_to_str(const ndarray<const int64_t,6> &a, int indent=0, int lvl=0);
std::string ndarray_uint8_1_to_str(const ndarray<const uint8_t,1> &a, int indent=0, int lvl=0);
std::string ndarray_uint8_2_to_str(const ndarray<const uint8_t,2> &a, int indent=0, int lvl=0);
std::string ndarray_uint8_3_to_str(const ndarray<const uint8_t,3> &a, int indent=0, int lvl=0);
std::string ndarray_uint8_4_to_str(const ndarray<const uint8_t,4> &a, int indent=0, int lvl=0);
std::string ndarray_uint8_5_to_str(const ndarray<const uint8_t,5> &a, int indent=0, int lvl=0);
std::string ndarray_uint8_6_to_str(const ndarray<const uint8_t,6> &a, int indent=0, int lvl=0);

} // namespace psana_test

#endif
