#ifndef PSANA_GSC16AI_DDL_H
#define PSANA_GSC16AI_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include <vector>
#include <iosfwd>
#include <cstring>
#include "ndarray/ndarray.h"
#include "pdsdata/xtc/TypeId.hh"
namespace Psana {
namespace Gsc16ai {

/** @class ConfigV1

  
*/


class ConfigV1 {
public:
  enum { TypeId = Pds::TypeId::Id_Gsc16aiConfig /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  enum { LowestChannel = 0 };
  enum { HighestChannel = 15 };
  enum { LowestFps = 1 };
  enum { HighestFps = 120 };
  enum InputMode {
    InputMode_Differential = 0,
    InputMode_Zero = 1,
    InputMode_Vref = 2,
  };
  enum VoltageRange {
    VoltageRange_10V = 0,
    VoltageRange_5V,
    VoltageRange_2_5V,
  };
  enum TriggerMode {
    TriggerMode_ExtPos = 0,
    TriggerMode_ExtNeg,
    TriggerMode_IntClk,
  };
  enum DataFormat {
    DataFormat_TwosComplement = 0,
    DataFormat_OffsetBinary,
  };
  virtual ~ConfigV1();
  virtual Gsc16ai::ConfigV1::VoltageRange voltageRange() const = 0;
  virtual uint16_t firstChan() const = 0;
  virtual uint16_t lastChan() const = 0;
  virtual Gsc16ai::ConfigV1::InputMode inputMode() const = 0;
  virtual Gsc16ai::ConfigV1::TriggerMode triggerMode() const = 0;
  virtual Gsc16ai::ConfigV1::DataFormat dataFormat() const = 0;
  virtual uint16_t fps() const = 0;
  virtual uint8_t autocalibEnable() const = 0;
  virtual uint8_t timeTagEnable() const = 0;
  virtual uint16_t numChannels() const = 0;
};
std::ostream& operator<<(std::ostream& str, Gsc16ai::ConfigV1::InputMode enval);
std::ostream& operator<<(std::ostream& str, Gsc16ai::ConfigV1::VoltageRange enval);
std::ostream& operator<<(std::ostream& str, Gsc16ai::ConfigV1::TriggerMode enval);
std::ostream& operator<<(std::ostream& str, Gsc16ai::ConfigV1::DataFormat enval);

/** @class DataV1

  
*/

class ConfigV1;

class DataV1 {
public:
  enum { TypeId = Pds::TypeId::Id_Gsc16aiData /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  virtual ~DataV1();
  virtual ndarray<const uint16_t, 1> timestamp() const = 0;
  virtual ndarray<const uint16_t, 1> channelValue() const = 0;
};
} // namespace Gsc16ai
} // namespace Psana
#endif // PSANA_GSC16AI_DDL_H
