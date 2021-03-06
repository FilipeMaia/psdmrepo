#ifndef PSANA_USDUSB_DDL_H
#define PSANA_USDUSB_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include <vector>
#include <iosfwd>
#include <cstring>
#include "ndarray/ndarray.h"
#include "pdsdata/xtc/TypeId.hh"
namespace Psana {
namespace UsdUsb {

/** @class ConfigV1

  
*/


class ConfigV1 {
public:
  enum { TypeId = Pds::TypeId::Id_UsdUsbConfig /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  enum { NCHANNELS = 4 };
  enum Count_Mode {
    WRAP_FULL,
    LIMIT,
    HALT,
    WRAP_PRESET,
  };
  enum Quad_Mode {
    CLOCK_DIR,
    X1,
    X2,
    X4,
  };
  virtual ~ConfigV1();
  virtual ndarray<const uint32_t, 1> counting_mode() const = 0;
  virtual ndarray<const uint32_t, 1> quadrature_mode() const = 0;
};
std::ostream& operator<<(std::ostream& str, UsdUsb::ConfigV1::Count_Mode enval);
std::ostream& operator<<(std::ostream& str, UsdUsb::ConfigV1::Quad_Mode enval);

/** @class DataV1

  
*/


class DataV1 {
public:
  enum { TypeId = Pds::TypeId::Id_UsdUsbData /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  enum { Encoder_Inputs = 4 };
  enum { Analog_Inputs = 4 };
  enum { Digital_Inputs = 8 };
  virtual ~DataV1();
  virtual uint8_t digital_in() const = 0;
  virtual uint32_t timestamp() const = 0;
  virtual ndarray<const uint8_t, 1> status() const = 0;
  virtual ndarray<const uint16_t, 1> analog_in() const = 0;
  /** Return lower 24 bits of _count array as signed integer values. */
  virtual ndarray<const int32_t, 1> encoder_count() const = 0;
};
} // namespace UsdUsb
} // namespace Psana
#endif // PSANA_USDUSB_DDL_H
