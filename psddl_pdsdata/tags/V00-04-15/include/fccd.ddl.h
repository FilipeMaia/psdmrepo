#ifndef PSDDLPDS_FCCD_DDL_H
#define PSDDLPDS_FCCD_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include <vector>
#include <cstddef>
#include "pdsdata/xtc/TypeId.hh"
#include "ndarray/ndarray.h"
namespace PsddlPds {
namespace FCCD {

/** @class FccdConfigV1

  
*/


class FccdConfigV1 {
public:
  enum { TypeId = Pds::TypeId::Id_FccdConfig /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  enum { Row_Pixels = 500 };
  enum { Column_Pixels = 576 };
  enum { Trimmed_Row_Pixels = 480 };
  enum { Trimmed_Column_Pixels = 480 };
  enum Depth {
    Sixteen_bit = 16,
  };
  enum Output_Source {
    Output_FIFO = 0,
    Output_Pattern4 = 4,
  };
  uint16_t outputMode() const { return _u16OutputMode; }
  uint32_t width() const;
  uint32_t height() const;
  uint32_t trimmedWidth() const;
  uint32_t trimmedHeight() const;
  static uint32_t _sizeof()  { return 2; }
private:
  uint16_t	_u16OutputMode;
};

/** @class FccdConfigV2

  
*/

#pragma pack(push,4)

class FccdConfigV2 {
public:
  enum { TypeId = Pds::TypeId::Id_FccdConfig /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 2 /**< XTC type version number */ };
  enum { Row_Pixels = 500 };
  enum { Column_Pixels = 576 * 2 };
  enum { Trimmed_Row_Pixels = 480 };
  enum { Trimmed_Column_Pixels = 480 };
  enum { NVoltages = 17 };
  enum { NWaveforms = 15 };
  enum Depth {
    Eight_bit = 8,
    Sixteen_bit = 16,
  };
  enum Output_Source {
    Output_FIFO = 0,
    Test_Pattern1 = 1,
    Test_Pattern2 = 2,
    Test_Pattern3 = 3,
    Test_Pattern4 = 4,
  };
  uint16_t outputMode() const { return _outputMode; }
  uint8_t ccdEnable() const { return _ccdEnable; }
  uint8_t focusMode() const { return _focusMode; }
  uint32_t exposureTime() const { return _exposureTime; }
  ndarray<const float, 1> dacVoltages() const { return make_ndarray(&_dacVoltage[0], NVoltages); }
  ndarray<const uint16_t, 1> waveforms() const { return make_ndarray(&_waveform[0], NWaveforms); }
  uint32_t width() const;
  uint32_t height() const;
  uint32_t trimmedWidth() const;
  uint32_t trimmedHeight() const;
  static uint32_t _sizeof()  { return (((((8+(4*(NVoltages)))+(2*(NWaveforms)))+4)-1)/4)*4; }
private:
  uint16_t	_outputMode;
  uint8_t	_ccdEnable;
  uint8_t	_focusMode;
  uint32_t	_exposureTime;
  float	_dacVoltage[NVoltages];
  uint16_t	_waveform[NWaveforms];
};
#pragma pack(pop)
} // namespace FCCD
} // namespace PsddlPds
#endif // PSDDLPDS_FCCD_DDL_H
