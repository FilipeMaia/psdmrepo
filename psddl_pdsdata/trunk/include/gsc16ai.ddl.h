#ifndef PSDDLPDS_GSC16AI_DDL_H
#define PSDDLPDS_GSC16AI_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include "pdsdata/xtc/TypeId.hh"

#include <vector>

#include <cstddef>

namespace PsddlPds {
namespace Gsc16ai {

/** @class ConfigV1

  
*/

#pragma pack(push,4)

class ConfigV1 {
public:
  enum {
    Version = 1 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_Gsc16aiConfig /**< XTC type ID value (from Pds::TypeId class) */
  };
  enum {
    LowestChannel = 0 /**<  */
  };
  enum {
    HighestChannel = 15 /**<  */
  };
  enum {
    LowestFps = 1 /**<  */
  };
  enum {
    HighestFps = 120 /**<  */
  };
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
  uint16_t voltageRange() const {return _voltageRange;}
  uint16_t firstChan() const {return _firstChan;}
  uint16_t lastChan() const {return _lastChan;}
  uint16_t inputMode() const {return _inputMode;}
  uint16_t triggerMode() const {return _triggerMode;}
  uint16_t dataFormat() const {return _dataFormat;}
  uint16_t fps() const {return _fps;}
  uint8_t autocalibEnable() const {return _autocalibEnable;}
  uint8_t timeTagEnable() const {return _timeTagEnable;}
  uint16_t numChannels() const {return this->_lastChan - this->_firstChan + 1;}
  static uint32_t _sizeof()  {return 16;}
private:
  uint16_t	_voltageRange;
  uint16_t	_firstChan;
  uint16_t	_lastChan;
  uint16_t	_inputMode;
  uint16_t	_triggerMode;
  uint16_t	_dataFormat;
  uint16_t	_fps;
  uint8_t	_autocalibEnable;
  uint8_t	_timeTagEnable;
};
#pragma pack(pop)

/** @class DataV1

  
*/

class ConfigV1;

class DataV1 {
public:
  enum {
    Version = 1 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_Gsc16aiData /**< XTC type ID value (from Pds::TypeId class) */
  };
  const uint16_t* timestamp() const {return &_timestamp[0];}
  const uint16_t* channelValue() const {
    ptrdiff_t offset=6;
    return (const uint16_t*)(((const char*)this)+offset);
  }
  static uint32_t _sizeof(const Gsc16ai::ConfigV1& cfg)  {return (0+(2*(3)))+(2*(cfg.numChannels()));}
  /** Method which returns the shape (dimensions) of the data returned by timestamp() method. */
  std::vector<int> timestamp_shape() const;
  /** Method which returns the shape (dimensions) of the data returned by channelValue() method. */
  std::vector<int> channelValue_shape(const Gsc16ai::ConfigV1& cfg) const;
private:
  uint16_t	_timestamp[3];
  //uint16_t	_channelValue[cfg.numChannels()];
};
} // namespace Gsc16ai
} // namespace PsddlPds
#endif // PSDDLPDS_GSC16AI_DDL_H
