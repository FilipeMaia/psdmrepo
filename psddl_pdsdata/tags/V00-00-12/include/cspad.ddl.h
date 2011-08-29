#ifndef PSDDLPDS_CSPAD_DDL_H
#define PSDDLPDS_CSPAD_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include "pdsdata/xtc/TypeId.hh"

#include <vector>

#include <cstddef>

namespace PsddlPds {
namespace CsPad {
  enum {
    MaxQuadsPerSensor = 4 /**< Defines number of quadrants in a CsPad device. */
  };
  enum {
    ASICsPerQuad = 16 /**< Total number of ASICs in one quadrant. */
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
    TwoByTwosPerQuad = 4 /**< Total number of 2x2s in single quadrant. */
  };
  enum {
    SectorsPerQuad = 8 /**< Total number of sectors (2x1) per single quadrant. */
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

/** @class CsPadDigitalPotsCfg

  Class defining configuration for CsPad POTs?
*/


class CsPadDigitalPotsCfg {
public:
  const uint8_t* pots() const {return &_pots[0];}
  static uint32_t _sizeof()  {return 0+(1*(PotsPerQuad));}
  /** Method which returns the shape (dimensions) of the data returned by pots() method. */
  std::vector<int> pots_shape() const;
private:
  uint8_t	_pots[PotsPerQuad];
};

/** @class CsPadReadOnlyCfg

  Class defining read-only configuration.
*/


class CsPadReadOnlyCfg {
public:
  uint32_t shiftTest() const {return _shiftTest;}
  uint32_t version() const {return _version;}
  static uint32_t _sizeof()  {return 8;}
private:
  uint32_t	_shiftTest;
  uint32_t	_version;
};

/** @class ProtectionSystemThreshold

  
*/


class ProtectionSystemThreshold {
public:
  uint32_t adcThreshold() const {return _adcThreshold;}
  uint32_t pixelCountThreshold() const {return _pixelCountThreshold;}
  static uint32_t _sizeof()  {return 8;}
private:
  uint32_t	_adcThreshold;
  uint32_t	_pixelCountThreshold;
};

/** @class CsPadGainMapCfg

  Class defining ASIC gain map.
*/


class CsPadGainMapCfg {
public:
  /** Array with the gain map for single ASIC. */
  const uint16_t* gainMap() const {return &_gainMap[0][0];}
  static uint32_t _sizeof()  {return 0+(2*(ColumnsPerASIC)*(MaxRowsPerASIC));}
  /** Method which returns the shape (dimensions) of the data returned by gainMap() method. */
  std::vector<int> gainMap_shape() const;
private:
  uint16_t	_gainMap[ColumnsPerASIC][MaxRowsPerASIC];	/**< Array with the gain map for single ASIC. */
};

/** @class ConfigV1QuadReg

  Configuration data for single quadrant.
*/


class ConfigV1QuadReg {
public:
  const uint32_t* shiftSelect() const {return &_shiftSelect[0];}
  const uint32_t* edgeSelect() const {return &_edgeSelect[0];}
  uint32_t readClkSet() const {return _readClkSet;}
  uint32_t readClkHold() const {return _readClkHold;}
  uint32_t dataMode() const {return _dataMode;}
  uint32_t prstSel() const {return _prstSel;}
  uint32_t acqDelay() const {return _acqDelay;}
  uint32_t intTime() const {return _intTime;}
  uint32_t digDelay() const {return _digDelay;}
  uint32_t ampIdle() const {return _ampIdle;}
  uint32_t injTotal() const {return _injTotal;}
  uint32_t rowColShiftPer() const {return _rowColShiftPer;}
  /** read-only configuration */
  const CsPad::CsPadReadOnlyCfg& ro() const {return _readOnly;}
  const CsPad::CsPadDigitalPotsCfg& dp() const {return _digitalPots;}
  /** Gain map. */
  const CsPad::CsPadGainMapCfg& gm() const {return _gainMap;}
  static uint32_t _sizeof()  {return ((((((((((((((0+(4*(TwoByTwosPerQuad)))+(4*(TwoByTwosPerQuad)))+4)+4)+4)+4)+4)+4)+4)+4)+4)+4)+(CsPad::CsPadReadOnlyCfg::_sizeof()))+(CsPad::CsPadDigitalPotsCfg::_sizeof()))+(CsPad::CsPadGainMapCfg::_sizeof());}
  /** Method which returns the shape (dimensions) of the data returned by shiftSelect() method. */
  std::vector<int> shiftSelect_shape() const;
  /** Method which returns the shape (dimensions) of the data returned by edgeSelect() method. */
  std::vector<int> edgeSelect_shape() const;
private:
  uint32_t	_shiftSelect[TwoByTwosPerQuad];
  uint32_t	_edgeSelect[TwoByTwosPerQuad];
  uint32_t	_readClkSet;
  uint32_t	_readClkHold;
  uint32_t	_dataMode;
  uint32_t	_prstSel;
  uint32_t	_acqDelay;
  uint32_t	_intTime;
  uint32_t	_digDelay;
  uint32_t	_ampIdle;
  uint32_t	_injTotal;
  uint32_t	_rowColShiftPer;
  CsPad::CsPadReadOnlyCfg	_readOnly;	/**< read-only configuration */
  CsPad::CsPadDigitalPotsCfg	_digitalPots;
  CsPad::CsPadGainMapCfg	_gainMap;	/**< Gain map. */
};

/** @class ConfigV1

  Configuration data for complete CsPad device.
*/


class ConfigV1 {
public:
  enum {
    Version = 1 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_CspadConfig /**< XTC type ID value (from Pds::TypeId class) */
  };
  uint32_t concentratorVersion() const {return _concentratorVersion;}
  uint32_t runDelay() const {return _runDelay;}
  uint32_t eventCode() const {return _eventCode;}
  uint32_t inactiveRunMode() const {return _inactiveRunMode;}
  uint32_t activeRunMode() const {return _activeRunMode;}
  uint32_t tdi() const {return _testDataIndex;}
  uint32_t payloadSize() const {return _payloadPerQuad;}
  uint32_t badAsicMask0() const {return _badAsicMask0;}
  uint32_t badAsicMask1() const {return _badAsicMask1;}
  uint32_t asicMask() const {return _AsicMask;}
  uint32_t quadMask() const {return _quadMask;}
  const CsPad::ConfigV1QuadReg& quads(uint32_t i0) const {return _quads[i0];}
  uint32_t numAsicsRead() const;
  uint32_t numQuads() const;
  uint32_t numSect() const;
  static uint32_t _sizeof()  {return 44+(CsPad::ConfigV1QuadReg::_sizeof()*(MaxQuadsPerSensor));}
  /** Method which returns the shape (dimensions) of the data returned by quads() method. */
  std::vector<int> quads_shape() const;
private:
  uint32_t	_concentratorVersion;
  uint32_t	_runDelay;
  uint32_t	_eventCode;
  uint32_t	_inactiveRunMode;
  uint32_t	_activeRunMode;
  uint32_t	_testDataIndex;
  uint32_t	_payloadPerQuad;
  uint32_t	_badAsicMask0;
  uint32_t	_badAsicMask1;
  uint32_t	_AsicMask;
  uint32_t	_quadMask;
  CsPad::ConfigV1QuadReg	_quads[MaxQuadsPerSensor];
};

/** @class ConfigV2

  Configuration data for complete CsPad device.
*/


class ConfigV2 {
public:
  enum {
    Version = 2 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_CspadConfig /**< XTC type ID value (from Pds::TypeId class) */
  };
  uint32_t concentratorVersion() const {return _concentratorVersion;}
  uint32_t runDelay() const {return _runDelay;}
  uint32_t eventCode() const {return _eventCode;}
  uint32_t inactiveRunMode() const {return _inactiveRunMode;}
  uint32_t activeRunMode() const {return _activeRunMode;}
  uint32_t tdi() const {return _testDataIndex;}
  uint32_t payloadSize() const {return _payloadPerQuad;}
  uint32_t badAsicMask0() const {return _badAsicMask0;}
  uint32_t badAsicMask1() const {return _badAsicMask1;}
  uint32_t asicMask() const {return _AsicMask;}
  uint32_t quadMask() const {return _quadMask;}
  const CsPad::ConfigV1QuadReg& quads(uint32_t i0) const {return _quads[i0];}
  uint32_t numAsicsRead() const;
  /** ROI mask for given quadrant */
  uint32_t roiMask(uint32_t iq) const;
  /** Number of ASICs in given quadrant */
  uint32_t numAsicsStored(uint32_t iq) const;
  /** Total number of quadrants in setup */
  uint32_t numQuads() const;
  /** Total number of sections (2x1) in all quadrants */
  uint32_t numSect() const;
  static uint32_t _sizeof()  {return 48+(CsPad::ConfigV1QuadReg::_sizeof()*(MaxQuadsPerSensor));}
  /** Method which returns the shape (dimensions) of the data returned by quads() method. */
  std::vector<int> quads_shape() const;
private:
  uint32_t	_concentratorVersion;
  uint32_t	_runDelay;
  uint32_t	_eventCode;
  uint32_t	_inactiveRunMode;
  uint32_t	_activeRunMode;
  uint32_t	_testDataIndex;
  uint32_t	_payloadPerQuad;
  uint32_t	_badAsicMask0;
  uint32_t	_badAsicMask1;
  uint32_t	_AsicMask;
  uint32_t	_quadMask;
  uint32_t	_roiMask;
  CsPad::ConfigV1QuadReg	_quads[MaxQuadsPerSensor];
};

/** @class ConfigV3

  Configuration data for complete CsPad device.
*/


class ConfigV3 {
public:
  enum {
    Version = 3 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_CspadConfig /**< XTC type ID value (from Pds::TypeId class) */
  };
  uint32_t concentratorVersion() const {return _concentratorVersion;}
  uint32_t runDelay() const {return _runDelay;}
  uint32_t eventCode() const {return _eventCode;}
  const CsPad::ProtectionSystemThreshold& protectionThresholds(uint32_t i0) const {return _protectionThresholds[i0];}
  uint32_t protectionEnable() const {return _protectionEnable;}
  uint32_t inactiveRunMode() const {return _inactiveRunMode;}
  uint32_t activeRunMode() const {return _activeRunMode;}
  uint32_t tdi() const {return _testDataIndex;}
  uint32_t payloadSize() const {return _payloadPerQuad;}
  uint32_t badAsicMask0() const {return _badAsicMask0;}
  uint32_t badAsicMask1() const {return _badAsicMask1;}
  uint32_t asicMask() const {return _AsicMask;}
  uint32_t quadMask() const {return _quadMask;}
  const CsPad::ConfigV1QuadReg& quads(uint32_t i0) const {return _quads[i0];}
  uint32_t numAsicsRead() const;
  /** ROI mask for given quadrant */
  uint32_t roiMask(uint32_t iq) const;
  /** Number of ASICs in given quadrant */
  uint32_t numAsicsStored(uint32_t iq) const;
  /** Total number of quadrants in setup */
  uint32_t numQuads() const;
  /** Total number of sections (2x1) in all quadrants */
  uint32_t numSect() const;
  static uint32_t _sizeof()  {return (((((((((((12+(CsPad::ProtectionSystemThreshold::_sizeof()*(MaxQuadsPerSensor)))+4)+4)+4)+4)+4)+4)+4)+4)+4)+4)+(CsPad::ConfigV1QuadReg::_sizeof()*(MaxQuadsPerSensor));}
  /** Method which returns the shape (dimensions) of the data returned by protectionThresholds() method. */
  std::vector<int> protectionThresholds_shape() const;
  /** Method which returns the shape (dimensions) of the data returned by quads() method. */
  std::vector<int> quads_shape() const;
private:
  uint32_t	_concentratorVersion;
  uint32_t	_runDelay;
  uint32_t	_eventCode;
  CsPad::ProtectionSystemThreshold	_protectionThresholds[MaxQuadsPerSensor];
  uint32_t	_protectionEnable;
  uint32_t	_inactiveRunMode;
  uint32_t	_activeRunMode;
  uint32_t	_testDataIndex;
  uint32_t	_payloadPerQuad;
  uint32_t	_badAsicMask0;
  uint32_t	_badAsicMask1;
  uint32_t	_AsicMask;
  uint32_t	_quadMask;
  uint32_t	_roiMask;
  CsPad::ConfigV1QuadReg	_quads[MaxQuadsPerSensor];
};

/** @class ElementV1

  CsPad data from single CsPad quadrant.
*/

class ConfigV1;
class ConfigV2;

class ElementV1 {
public:
  enum {
    Nsbtemp = 4 /**< Number of the elements in _sbtemp array. */
  };
  /** Virtual channel number. */
  uint32_t virtual_channel() const {return uint32_t(this->_word0 & 0x3);}
  /** Lane number. */
  uint32_t lane() const {return uint32_t((this->_word0>>6) & 0x3);}
  uint32_t tid() const {return uint32_t((this->_word0>>8) & 0xffffff);}
  uint32_t acq_count() const {return uint32_t(this->_word1 & 0xffff);}
  uint32_t op_code() const {return uint32_t((this->_word1>>16) & 0xff);}
  /** Quadrant number. */
  uint32_t quad() const {return uint32_t((this->_word1>>24) & 0x3);}
  /** Counter incremented on every event. */
  uint32_t seq_count() const {return _seq_count;}
  uint32_t ticks() const {return _ticks;}
  uint32_t fiducials() const {return _fiducials;}
  const uint16_t* sb_temp() const {return &_sbtemp[0];}
  uint32_t frame_type() const {return _frame_type;}
  const uint16_t* data() const {
    ptrdiff_t offset=32;
    return (const uint16_t*)(((const char*)this)+offset);
  }
  static uint32_t _sizeof(const CsPad::ConfigV1& cfg)  {return (((20+(2*(Nsbtemp)))+4)+(2*(cfg.numAsicsRead()/2)*( ColumnsPerASIC)*( MaxRowsPerASIC*2)))+(2*(2));}
  static uint32_t _sizeof(const CsPad::ConfigV2& cfg)  {return (((20+(2*(Nsbtemp)))+4)+(2*(cfg.numAsicsRead()/2)*( ColumnsPerASIC)*( MaxRowsPerASIC*2)))+(2*(2));}
  /** Method which returns the shape (dimensions) of the data returned by sb_temp() method. */
  std::vector<int> sb_temp_shape() const;
  /** Method which returns the shape (dimensions) of the data returned by data() method. */
  std::vector<int> data_shape(const CsPad::ConfigV1& cfg) const;
  /** Method which returns the shape (dimensions) of the data returned by data() method. */
  std::vector<int> data_shape(const CsPad::ConfigV2& cfg) const;
  /** Method which returns the shape (dimensions) of the data member _extra. */
  std::vector<int> _extra_shape() const;
private:
  uint32_t	_word0;
  uint32_t	_word1;
  uint32_t	_seq_count;	/**< Counter incremented on every event. */
  uint32_t	_ticks;
  uint32_t	_fiducials;
  uint16_t	_sbtemp[Nsbtemp];
  uint32_t	_frame_type;
  //uint16_t	_data[cfg.numAsicsRead()/2][ ColumnsPerASIC][ MaxRowsPerASIC*2];
  //uint16_t	_extra[2];
};

/** @class DataV1

  CsPad data from whole detector.
*/

class ConfigV1;
class ConfigV2;

class DataV1 {
public:
  enum {
    Version = 1 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_CspadElement /**< XTC type ID value (from Pds::TypeId class) */
  };
  /** Data objects, one element per quadrant. The size of the array is determined by 
            the numQuads() method of the configuration object. */
  const CsPad::ElementV1& quads(const CsPad::ConfigV1& cfg, uint32_t i0) const {
    ptrdiff_t offset=0;
    const CsPad::ElementV1* memptr = (const CsPad::ElementV1*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof(cfg);
    return *(const CsPad::ElementV1*)((const char*)memptr + (i0)*memsize);
  }
  /** Data objects, one element per quadrant. The size of the array is determined by 
            the numQuads() method of the configuration object. */
  const CsPad::ElementV1& quads(const CsPad::ConfigV2& cfg, uint32_t i0) const {
    ptrdiff_t offset=0;
    const CsPad::ElementV1* memptr = (const CsPad::ElementV1*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof(cfg);
    return *(const CsPad::ElementV1*)((const char*)memptr + (i0)*memsize);
  }
  static uint32_t _sizeof(const CsPad::ConfigV1& cfg)  {return 0+(CsPad::ElementV1::_sizeof(cfg)*(cfg.numQuads()));}
  static uint32_t _sizeof(const CsPad::ConfigV2& cfg)  {return 0+(CsPad::ElementV1::_sizeof(cfg)*(cfg.numQuads()));}
  /** Method which returns the shape (dimensions) of the data returned by quads() method. */
  std::vector<int> quads_shape(const CsPad::ConfigV1& cfg) const;
  /** Method which returns the shape (dimensions) of the data returned by quads() method. */
  std::vector<int> quads_shape(const CsPad::ConfigV2& cfg) const;
private:
  //CsPad::ElementV1	_quads[cfg.numQuads()];
};

/** @class ElementV2

  CsPad data from single CsPad quadrant.
*/

class ConfigV2;
class ConfigV3;

class ElementV2 {
public:
  enum {
    Nsbtemp = 4 /**< Number of the elements in _sbtemp array. */
  };
  /** Virtual channel number. */
  uint32_t virtual_channel() const {return uint32_t(this->_word0 & 0x3);}
  /** Lane number. */
  uint32_t lane() const {return uint32_t((this->_word0>>6) & 0x3);}
  uint32_t tid() const {return uint32_t((this->_word0>>8) & 0xffffff);}
  uint32_t acq_count() const {return uint32_t(this->_word1 & 0xffff);}
  uint32_t op_code() const {return uint32_t((this->_word1>>16) & 0xff);}
  /** Quadrant number. */
  uint32_t quad() const {return uint32_t((this->_word1>>24) & 0x3);}
  uint32_t seq_count() const {return _seq_count;}
  uint32_t ticks() const {return _ticks;}
  uint32_t fiducials() const {return _fiducials;}
  const uint16_t* sb_temp() const {return &_sbtemp[0];}
  uint32_t frame_type() const {return _frame_type;}
  const uint16_t* data() const {
    ptrdiff_t offset=32;
    return (const uint16_t*)(((const char*)this)+offset);
  }
  uint32_t _sizeof(const CsPad::ConfigV2& cfg) const {return (((20+(2*(Nsbtemp)))+4)+(2*(cfg.numAsicsStored(this->quad())/2)*( ColumnsPerASIC)*( MaxRowsPerASIC*2)))+(2*(2));}
  uint32_t _sizeof(const CsPad::ConfigV3& cfg) const {return (((20+(2*(Nsbtemp)))+4)+(2*(cfg.numAsicsStored(this->quad())/2)*( ColumnsPerASIC)*( MaxRowsPerASIC*2)))+(2*(2));}
  /** Method which returns the shape (dimensions) of the data returned by sb_temp() method. */
  std::vector<int> sb_temp_shape() const;
  /** Method which returns the shape (dimensions) of the data returned by data() method. */
  std::vector<int> data_shape(const CsPad::ConfigV2& cfg) const;
  /** Method which returns the shape (dimensions) of the data returned by data() method. */
  std::vector<int> data_shape(const CsPad::ConfigV3& cfg) const;
  /** Method which returns the shape (dimensions) of the data member _extra. */
  std::vector<int> _extra_shape() const;
private:
  uint32_t	_word0;
  uint32_t	_word1;
  uint32_t	_seq_count;
  uint32_t	_ticks;
  uint32_t	_fiducials;
  uint16_t	_sbtemp[Nsbtemp];
  uint32_t	_frame_type;
  //uint16_t	_data[cfg.numAsicsStored(this->quad())/2][ ColumnsPerASIC][ MaxRowsPerASIC*2];
  //uint16_t	_extra[2];
};

/** @class DataV2

  CsPad data from whole detector.
*/

class ConfigV2;
class ConfigV3;

class DataV2 {
public:
  enum {
    Version = 2 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_CspadElement /**< XTC type ID value (from Pds::TypeId class) */
  };
  /** Data objects, one element per quadrant. The size of the array is determined by 
            the numQuads() method of the configuration object. */
  const CsPad::ElementV2& quads(const CsPad::ConfigV2& cfg, uint32_t i0) const {
    const char* memptr = ((const char*)this)+0;
    for (uint32_t i=0; i != i0; ++ i) {
      memptr += ((const CsPad::ElementV2*)memptr)->_sizeof(cfg);
    }
    return *(const CsPad::ElementV2*)(memptr);
  }
  /** Data objects, one element per quadrant. The size of the array is determined by 
            the numQuads() method of the configuration object. */
  const CsPad::ElementV2& quads(const CsPad::ConfigV3& cfg, uint32_t i0) const {
    const char* memptr = ((const char*)this)+0;
    for (uint32_t i=0; i != i0; ++ i) {
      memptr += ((const CsPad::ElementV2*)memptr)->_sizeof(cfg);
    }
    return *(const CsPad::ElementV2*)(memptr);
  }
  /** Method which returns the shape (dimensions) of the data returned by quads() method. */
  std::vector<int> quads_shape(const CsPad::ConfigV2& cfg) const;
  /** Method which returns the shape (dimensions) of the data returned by quads() method. */
  std::vector<int> quads_shape(const CsPad::ConfigV3& cfg) const;
private:
  //CsPad::ElementV2	_quads[cfg.numQuads()];
};
} // namespace CsPad
} // namespace PsddlPds
#endif // PSDDLPDS_CSPAD_DDL_H
