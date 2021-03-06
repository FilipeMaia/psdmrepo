#ifndef PSDDLPDS_LUSI_DDL_H
#define PSDDLPDS_LUSI_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include <vector>
#include <iosfwd>
#include <cstddef>
#include "pdsdata/xtc/TypeId.hh"
#include "ndarray/ndarray.h"
namespace PsddlPds {
namespace Lusi {

/** @class DiodeFexConfigV1

  
*/

#pragma pack(push,4)

class DiodeFexConfigV1 {
public:
  enum { TypeId = Pds::TypeId::Id_DiodeFexConfig /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  enum { NRANGES = 3 };
  DiodeFexConfigV1()
  {
  }
  DiodeFexConfigV1(const float* arg__base, const float* arg__scale)
  {
    if (arg__base) std::copy(arg__base, arg__base+(3), &_base[0]);
    if (arg__scale) std::copy(arg__scale, arg__scale+(3), &_scale[0]);
  }
  /**     Note: this overloaded method accepts shared pointer argument which must point to an object containing
    this instance, the returned ndarray object can be used even after this instance disappears. */
  template <typename T>
  ndarray<const float, 1> base(const boost::shared_ptr<T>& owner) const { 
    const float* data = &_base[0];
    return make_ndarray(boost::shared_ptr<const float>(owner, data), NRANGES);
  }
  /**     Note: this method returns ndarray instance which does not control lifetime
    of the data, do not use returned ndarray after this instance disappears. */
  ndarray<const float, 1> base() const { return make_ndarray(&_base[0], NRANGES); }
  /**     Note: this overloaded method accepts shared pointer argument which must point to an object containing
    this instance, the returned ndarray object can be used even after this instance disappears. */
  template <typename T>
  ndarray<const float, 1> scale(const boost::shared_ptr<T>& owner) const { 
    const float* data = &_scale[0];
    return make_ndarray(boost::shared_ptr<const float>(owner, data), NRANGES);
  }
  /**     Note: this method returns ndarray instance which does not control lifetime
    of the data, do not use returned ndarray after this instance disappears. */
  ndarray<const float, 1> scale() const { return make_ndarray(&_scale[0], NRANGES); }
  static uint32_t _sizeof() { return (((((0+(4*(NRANGES)))+(4*(NRANGES)))+4)-1)/4)*4; }
private:
  float	_base[NRANGES];
  float	_scale[NRANGES];
};
#pragma pack(pop)

/** @class DiodeFexConfigV2

  
*/

#pragma pack(push,4)

class DiodeFexConfigV2 {
public:
  enum { TypeId = Pds::TypeId::Id_DiodeFexConfig /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 2 /**< XTC type version number */ };
  enum { NRANGES = 16 };
  DiodeFexConfigV2()
  {
  }
  DiodeFexConfigV2(const float* arg__base, const float* arg__scale)
  {
    if (arg__base) std::copy(arg__base, arg__base+(16), &_base[0]);
    if (arg__scale) std::copy(arg__scale, arg__scale+(16), &_scale[0]);
  }
  /**     Note: this overloaded method accepts shared pointer argument which must point to an object containing
    this instance, the returned ndarray object can be used even after this instance disappears. */
  template <typename T>
  ndarray<const float, 1> base(const boost::shared_ptr<T>& owner) const { 
    const float* data = &_base[0];
    return make_ndarray(boost::shared_ptr<const float>(owner, data), NRANGES);
  }
  /**     Note: this method returns ndarray instance which does not control lifetime
    of the data, do not use returned ndarray after this instance disappears. */
  ndarray<const float, 1> base() const { return make_ndarray(&_base[0], NRANGES); }
  /**     Note: this overloaded method accepts shared pointer argument which must point to an object containing
    this instance, the returned ndarray object can be used even after this instance disappears. */
  template <typename T>
  ndarray<const float, 1> scale(const boost::shared_ptr<T>& owner) const { 
    const float* data = &_scale[0];
    return make_ndarray(boost::shared_ptr<const float>(owner, data), NRANGES);
  }
  /**     Note: this method returns ndarray instance which does not control lifetime
    of the data, do not use returned ndarray after this instance disappears. */
  ndarray<const float, 1> scale() const { return make_ndarray(&_scale[0], NRANGES); }
  static uint32_t _sizeof() { return (((((0+(4*(NRANGES)))+(4*(NRANGES)))+4)-1)/4)*4; }
private:
  float	_base[NRANGES];
  float	_scale[NRANGES];
};
#pragma pack(pop)

/** @class DiodeFexV1

  
*/

#pragma pack(push,4)

class DiodeFexV1 {
public:
  enum { TypeId = Pds::TypeId::Id_DiodeFex /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  DiodeFexV1()
  {
  }
  DiodeFexV1(float arg__value)
    : _value(arg__value)
  {
  }
  float value() const { return _value; }
  static uint32_t _sizeof() { return 4; }
private:
  float	_value;
};
#pragma pack(pop)

/** @class IpmFexConfigV1

  
*/

#pragma pack(push,4)

class IpmFexConfigV1 {
public:
  enum { TypeId = Pds::TypeId::Id_IpmFexConfig /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  enum { NCHANNELS = 4 };
  IpmFexConfigV1() {}
  IpmFexConfigV1(const IpmFexConfigV1& other) {
    const char* src = reinterpret_cast<const char*>(&other);
    std::copy(src, src+other._sizeof(), reinterpret_cast<char*>(this));
  }
  IpmFexConfigV1& operator=(const IpmFexConfigV1& other) {
    const char* src = reinterpret_cast<const char*>(&other);
    std::copy(src, src+other._sizeof(), reinterpret_cast<char*>(this));
    return *this;
  }
  /**     Note: this overloaded method accepts shared pointer argument which must point to an object containing
    this instance, the returned ndarray object can be used even after this instance disappears. */
  template <typename T>
  ndarray<const Lusi::DiodeFexConfigV1, 1> diode(const boost::shared_ptr<T>& owner) const { 
    const Lusi::DiodeFexConfigV1* data = &_diode[0];
    return make_ndarray(boost::shared_ptr<const Lusi::DiodeFexConfigV1>(owner, data), NCHANNELS);
  }
  /**     Note: this method returns ndarray instance which does not control lifetime
    of the data, do not use returned ndarray after this instance disappears. */
  ndarray<const Lusi::DiodeFexConfigV1, 1> diode() const { return make_ndarray(&_diode[0], NCHANNELS); }
  float xscale() const { return _xscale; }
  float yscale() const { return _yscale; }
  static uint32_t _sizeof() { return ((((((0+(Lusi::DiodeFexConfigV1::_sizeof()*(NCHANNELS)))+4)+4)+4)-1)/4)*4; }
private:
  Lusi::DiodeFexConfigV1	_diode[NCHANNELS];
  float	_xscale;
  float	_yscale;
};
#pragma pack(pop)

/** @class IpmFexConfigV2

  
*/

#pragma pack(push,4)

class IpmFexConfigV2 {
public:
  enum { TypeId = Pds::TypeId::Id_IpmFexConfig /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 2 /**< XTC type version number */ };
  enum { NCHANNELS = 4 };
  IpmFexConfigV2() {}
  IpmFexConfigV2(const IpmFexConfigV2& other) {
    const char* src = reinterpret_cast<const char*>(&other);
    std::copy(src, src+other._sizeof(), reinterpret_cast<char*>(this));
  }
  IpmFexConfigV2& operator=(const IpmFexConfigV2& other) {
    const char* src = reinterpret_cast<const char*>(&other);
    std::copy(src, src+other._sizeof(), reinterpret_cast<char*>(this));
    return *this;
  }
  /**     Note: this overloaded method accepts shared pointer argument which must point to an object containing
    this instance, the returned ndarray object can be used even after this instance disappears. */
  template <typename T>
  ndarray<const Lusi::DiodeFexConfigV2, 1> diode(const boost::shared_ptr<T>& owner) const { 
    const Lusi::DiodeFexConfigV2* data = &_diode[0];
    return make_ndarray(boost::shared_ptr<const Lusi::DiodeFexConfigV2>(owner, data), NCHANNELS);
  }
  /**     Note: this method returns ndarray instance which does not control lifetime
    of the data, do not use returned ndarray after this instance disappears. */
  ndarray<const Lusi::DiodeFexConfigV2, 1> diode() const { return make_ndarray(&_diode[0], NCHANNELS); }
  float xscale() const { return _xscale; }
  float yscale() const { return _yscale; }
  static uint32_t _sizeof() { return ((((((0+(Lusi::DiodeFexConfigV2::_sizeof()*(NCHANNELS)))+4)+4)+4)-1)/4)*4; }
private:
  Lusi::DiodeFexConfigV2	_diode[NCHANNELS];
  float	_xscale;
  float	_yscale;
};
#pragma pack(pop)

/** @class IpmFexV1

  
*/

#pragma pack(push,4)

class IpmFexV1 {
public:
  enum { TypeId = Pds::TypeId::Id_IpmFex /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  enum { NCHANNELS = 4 };
  IpmFexV1()
  {
  }
  IpmFexV1(const float* arg__channel, float arg__sum, float arg__xpos, float arg__ypos)
    : _sum(arg__sum), _xpos(arg__xpos), _ypos(arg__ypos)
  {
    if (arg__channel) std::copy(arg__channel, arg__channel+(4), &_channel[0]);
  }
  /**     Note: this overloaded method accepts shared pointer argument which must point to an object containing
    this instance, the returned ndarray object can be used even after this instance disappears. */
  template <typename T>
  ndarray<const float, 1> channel(const boost::shared_ptr<T>& owner) const { 
    const float* data = &_channel[0];
    return make_ndarray(boost::shared_ptr<const float>(owner, data), NCHANNELS);
  }
  /**     Note: this method returns ndarray instance which does not control lifetime
    of the data, do not use returned ndarray after this instance disappears. */
  ndarray<const float, 1> channel() const { return make_ndarray(&_channel[0], NCHANNELS); }
  float sum() const { return _sum; }
  float xpos() const { return _xpos; }
  float ypos() const { return _ypos; }
  static uint32_t _sizeof() { return (((((((0+(4*(NCHANNELS)))+4)+4)+4)+4)-1)/4)*4; }
private:
  float	_channel[NCHANNELS];
  float	_sum;
  float	_xpos;
  float	_ypos;
};
#pragma pack(pop)

/** @class PimImageConfigV1

  
*/

#pragma pack(push,4)

class PimImageConfigV1 {
public:
  enum { TypeId = Pds::TypeId::Id_PimImageConfig /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  PimImageConfigV1()
  {
  }
  PimImageConfigV1(float arg__xscale, float arg__yscale)
    : _xscale(arg__xscale), _yscale(arg__yscale)
  {
  }
  float xscale() const { return _xscale; }
  float yscale() const { return _yscale; }
  static uint32_t _sizeof() { return 8; }
private:
  float	_xscale;
  float	_yscale;
};
#pragma pack(pop)
} // namespace Lusi
} // namespace PsddlPds
#endif // PSDDLPDS_LUSI_DDL_H
