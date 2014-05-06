#include "psana_test/dumpBasicTypes.h"

#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>

#include "psana_test/flattenNdarray.h"
#include "psana_test/adler.h"

using namespace std;
namespace {

std::vector<unsigned> getQuantileIndicies(unsigned n) {
  std::vector<unsigned> quantileIdx(5);
  quantileIdx[0]=0;
  quantileIdx[1]=std::max(0u,std::min(n-1,unsigned(double(n)/4.0)));
  quantileIdx[2]=std::max(0u,std::min(n-1,unsigned(double(n)/2.0)));
  quantileIdx[3]=std::max(0u,std::min(n-1,unsigned(double(3*n)/4.0)));
  quantileIdx[4]=std::max(0u,n-1);
  return quantileIdx;
}

template <class T, unsigned R>
string dump_ndarray(const ndarray<const T, R> &a) {
  ostringstream out;
  out << "dim=[ " << a.shape()[0];
  for (unsigned idx = 1; idx < R; ++idx) {
    out << " x " << a.shape()[idx];
  }
  std::vector<T> a_flat;
  psana_test::flatten(a, a_flat);
  uint64_t sizeBytes = a_flat.size() * sizeof(T);
  uint64_t adler = psana_test_adler32((const void *)(& (a_flat.at(0)) ), sizeBytes);
  out << " ] adler32=" << psana_test::dump_uint64(adler);
  std::sort(a_flat.begin(), a_flat.end());
  vector<unsigned> quantileIdx = getQuantileIndicies(a_flat.size());
  out << " min=" << a_flat.at( quantileIdx.at(0) )
      << " 25th=" << a_flat.at( quantileIdx.at(1) )
      << " median=" << a_flat.at( quantileIdx.at(2) )
      << " 75th=" << a_flat.at( quantileIdx.at(3) )
      << " max=" << a_flat.at( quantileIdx.at(4) );
  return out.str();
}


} // local namespace

// *********************************************************************
// public functions - dump functions

string psana_test::dump_int8(int8_t v) {
  ostringstream out;
  out << v;
  return out.str();
}

string psana_test::dump_int16(int16_t v) {
  ostringstream out;
  out << v;
  return out.str();
}

string psana_test::dump_int32(int32_t v) {
  ostringstream out;
  out << v;
  return out.str();
}

string psana_test::dump_int64(int64_t v) {
  ostringstream out;
  out << v;
  return out.str();
}

string psana_test::dump_uint8(uint8_t v) {
  ostringstream out;
  out << hex << v;
  return out.str();
}

string psana_test::dump_uint16(uint16_t v) {
  ostringstream out;
  out << hex << v;
  return out.str();
}

string psana_test::dump_uint32(uint32_t v) {
  ostringstream out;
  out << hex << v;
  return out.str();
}

string psana_test::dump_uint64(uint64_t v) {
  ostringstream out;
  out << hex << v;
  return out.str();
}

string psana_test::dump_str(const char *v) {
  ostringstream out;
  out << string(v);
  return out.str();
}

string psana_test::dump_float(float v) {
  ostringstream out;
  out << scientific << v;
  return out.str();
}

string psana_test::dump_double(double v) {
  ostringstream out;
  out << scientific << setprecision(4) << v;
  return out.str();
}

std::string psana_test::dump_ndarray_float32_1(const ndarray<const float,1> &a) {
  return string("ndarray_float32_1: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_float32_2(const ndarray<const float,2> &a) {
  return string("ndarray_float32_2: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_float32_3(const ndarray<const float,3> &a) {
  return string("ndarray_float32_3: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_float32_4(const ndarray<const float,4> &a) {
  return string("ndarray_float32_4: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_float32_5(const ndarray<const float,5> &a) {
  return string("ndarray_float32_5: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_float32_6(const ndarray<const float,6> &a) {
  return string("ndarray_float32_6: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_float64_1(const ndarray<const double,1> &a) {
  return string("ndarray_float64_1: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_float64_2(const ndarray<const double,2> &a) {
  return string("ndarray_float64_2: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_float64_3(const ndarray<const double,3> &a) {
  return string("ndarray_float64_3: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_float64_4(const ndarray<const double,4> &a) {
  return string("ndarray_float64_4: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_float64_5(const ndarray<const double,5> &a) {
  return string("ndarray_float64_5: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_float64_6(const ndarray<const double,6> &a) {
  return string("ndarray_float64_6: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_int16_1(const ndarray<const int16_t,1> &a) {
  return string("ndarray_int16_1: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_int16_2(const ndarray<const int16_t,2> &a) {
  return string("ndarray_int16_2: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_int16_3(const ndarray<const int16_t,3> &a) {
  return string("ndarray_int16_3: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_int16_4(const ndarray<const int16_t,4> &a) {
  return string("ndarray_int16_4: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_int16_5(const ndarray<const int16_t,5> &a) {
  return string("ndarray_int16_5: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_int16_6(const ndarray<const int16_t,6> &a) {
  return string("ndarray_int16_6: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_int32_1(const ndarray<const int32_t,1> &a) {
  return string("ndarray_int32_1: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_int32_2(const ndarray<const int32_t,2> &a) {
  return string("ndarray_int32_2: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_int32_3(const ndarray<const int32_t,3> &a) {
  return string("ndarray_int32_3: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_int32_4(const ndarray<const int32_t,4> &a) {
  return string("ndarray_int32_4: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_int32_5(const ndarray<const int32_t,5> &a) {
  return string("ndarray_int32_5: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_int32_6(const ndarray<const int32_t,6> &a) {
  return string("ndarray_int32_6: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_int64_1(const ndarray<const int64_t,1> &a) {
  return string("ndarray_int64_1: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_int64_2(const ndarray<const int64_t,2> &a) {
  return string("ndarray_int64_2: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_int64_3(const ndarray<const int64_t,3> &a) {
  return string("ndarray_int64_3: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_int64_4(const ndarray<const int64_t,4> &a) {
  return string("ndarray_int64_4: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_int64_5(const ndarray<const int64_t,5> &a) {
  return string("ndarray_int64_5: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_int64_6(const ndarray<const int64_t,6> &a) {
  return string("ndarray_int64_6: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_uint8_1(const ndarray<const uint8_t,1> &a) {
  return string("ndarray_uint8_1: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_uint8_2(const ndarray<const uint8_t,2> &a) {
  return string("ndarray_uint8_2: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_uint8_3(const ndarray<const uint8_t,3> &a) {
  return string("ndarray_uint8_3: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_uint8_4(const ndarray<const uint8_t,4> &a) {
  return string("ndarray_uint8_4: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_uint8_5(const ndarray<const uint8_t,5> &a) {
  return string("ndarray_uint8_5: ") + dump_ndarray(a);
}

std::string psana_test::dump_ndarray_uint8_6(const ndarray<const uint8_t,6> &a) {
  return string("ndarray_uint8_6: ") + dump_ndarray(a);
}






