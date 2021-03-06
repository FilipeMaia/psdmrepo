#ifndef PSANA_CSPAD2X2_DDL_H
#define PSANA_CSPAD2X2_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include <vector>
#include "ndarray/ndarray.h"
#include "pdsdata/xtc/TypeId.hh"
namespace Psana {
namespace CsPad2x2 {
  enum {
    QuadsPerSensor = 1 /**< Defines number of quadrants in a CsPad2x2 device. */
  };
  enum {
    ASICsPerQuad = 4 /**< Total number of ASICs in one quadrant. */
  };
  enum {
    RowsPerBank = 26 /**< Number of rows per readout bank? */
  };
  enum {
    FullBanksPerASIC = 7 /**< Number of full readout banks per one ASIC? */
  };
  enum {
    BanksPerASIC = 8 /**< Number of readout banks per one ASIC? */
  };
  enum {
    ColumnsPerASIC = 185 /**< Number of columns readout by single ASIC. */
  };
  enum {
    MaxRowsPerASIC = 194 /**< Maximum number of rows readout by single ASIC. */
  };
  enum {
    PotsPerQuad = 80 /**< Number of POTs? per single quadrant. */
  };
  enum {
    TwoByTwosPerQuad = 1 /**< Total number of 2x2s in single quadrant. */
  };
  enum {
    SectorsPerQuad = 2 /**< Total number of sectors (2x1) per single quadrant. */
  };

  /** Enum specifying different running modes. */
  enum RunModes {
    NoRunning,
    RunButDrop,
    RunAndSendToRCE,
    RunAndSendTriggeredByTTL,
    ExternalTriggerSendToRCE,
    ExternalTriggerDrop,
    NumberOfRunModes,
  };

  /** Enum specifying different data collection modes. */
  enum DataModes {
    normal = 0,
    shiftTest = 1,
    testData = 2,
    reserved = 3,
  };

/** @class CsPad2x2DigitalPotsCfg

  Class defining configuration for CsPad POTs?
*/


class CsPad2x2DigitalPotsCfg {
public:
  virtual ~CsPad2x2DigitalPotsCfg();
  virtual ndarray<uint8_t, 1> pots() const = 0;
};

/** @class CsPad2x2ReadOnlyCfg

  Class defining read-only configuration.
*/


class CsPad2x2ReadOnlyCfg {
public:
  virtual ~CsPad2x2ReadOnlyCfg();
  virtual uint32_t shiftTest() const = 0;
  virtual uint32_t version() const = 0;
};

/** @class ProtectionSystemThreshold

  
*/


class ProtectionSystemThreshold {
public:
  virtual ~ProtectionSystemThreshold();
  virtual uint32_t adcThreshold() const = 0;
  virtual uint32_t pixelCountThreshold() const = 0;
};

/** @class CsPad2x2GainMapCfg

  Class defining ASIC gain map.
*/


class CsPad2x2GainMapCfg {
public:
  virtual ~CsPad2x2GainMapCfg();
  /** Array with the gain map for single ASIC. */
  virtual ndarray<uint16_t, 2> gainMap() const = 0;
};

/** @class ConfigV1QuadReg

  Configuration data for single quadrant.
*/


class ConfigV1QuadReg {
public:
  virtual ~ConfigV1QuadReg();
  virtual uint32_t shiftSelect() const = 0;
  virtual uint32_t edgeSelect() const = 0;
  virtual uint32_t readClkSet() const = 0;
  virtual uint32_t readClkHold() const = 0;
  virtual uint32_t dataMode() const = 0;
  virtual uint32_t prstSel() const = 0;
  virtual uint32_t acqDelay() const = 0;
  virtual uint32_t intTime() const = 0;
  virtual uint32_t digDelay() const = 0;
  virtual uint32_t ampIdle() const = 0;
  virtual uint32_t injTotal() const = 0;
  virtual uint32_t rowColShiftPer() const = 0;
  virtual uint32_t ampReset() const = 0;
  virtual uint32_t digCount() const = 0;
  virtual uint32_t digPeriod() const = 0;
  virtual uint32_t PeltierEnable() const = 0;
  virtual uint32_t kpConstant() const = 0;
  virtual uint32_t kiConstant() const = 0;
  virtual uint32_t kdConstant() const = 0;
  virtual uint32_t humidThold() const = 0;
  virtual uint32_t setPoint() const = 0;
  /** read-only configuration */
  virtual const CsPad2x2::CsPad2x2ReadOnlyCfg& ro() const = 0;
  virtual const CsPad2x2::CsPad2x2DigitalPotsCfg& dp() const = 0;
  /** Gain map. */
  virtual const CsPad2x2::CsPad2x2GainMapCfg& gm() const = 0;
};

/** @class ConfigV1

  Configuration data for complete CsPad device.
*/


class ConfigV1 {
public:
  enum { TypeId = Pds::TypeId::Id_Cspad2x2Config /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  virtual ~ConfigV1();
  virtual uint32_t concentratorVersion() const = 0;
  virtual const CsPad2x2::ProtectionSystemThreshold& protectionThreshold() const = 0;
  virtual uint32_t protectionEnable() const = 0;
  virtual uint32_t inactiveRunMode() const = 0;
  virtual uint32_t activeRunMode() const = 0;
  virtual uint32_t tdi() const = 0;
  virtual uint32_t payloadSize() const = 0;
  virtual uint32_t badAsicMask() const = 0;
  virtual uint32_t asicMask() const = 0;
  virtual uint32_t roiMask() const = 0;
  virtual const CsPad2x2::ConfigV1QuadReg& quad() const = 0;
  virtual uint32_t numAsicsRead() const = 0;
  /** Number of ASICs in given quadrant */
  virtual uint32_t numAsicsStored() const = 0;
};

/** @class ElementV1

  CsPad data from single 2x2 element.
*/


class ElementV1 {
public:
  enum { TypeId = Pds::TypeId::Id_Cspad2x2Element /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  enum { Nsbtemp = 4 /**< Number of the elements in _sbtemp array. */ };
  virtual ~ElementV1();
  /** Virtual channel number. */
  virtual uint32_t virtual_channel() const = 0;
  /** Lane number. */
  virtual uint32_t lane() const = 0;
  virtual uint32_t tid() const = 0;
  virtual uint32_t acq_count() const = 0;
  virtual uint32_t op_code() const = 0;
  /** Quadrant number. */
  virtual uint32_t quad() const = 0;
  virtual uint32_t seq_count() const = 0;
  virtual uint32_t ticks() const = 0;
  virtual uint32_t fiducials() const = 0;
  virtual ndarray<uint16_t, 1> sb_temp() const = 0;
  virtual uint32_t frame_type() const = 0;
  virtual ndarray<int16_t, 3> data() const = 0;
  /** Common mode value for a given section, section number can be 0 or 1.
                Will return 0 for data read from XTC, may be non-zero after calibration. */
  virtual float common_mode(uint32_t section) const = 0;
};
} // namespace CsPad2x2
} // namespace Psana
#endif // PSANA_CSPAD2X2_DDL_H
