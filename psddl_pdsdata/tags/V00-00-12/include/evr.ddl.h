#ifndef PSDDLPDS_EVR_DDL_H
#define PSDDLPDS_EVR_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include "pdsdata/xtc/TypeId.hh"

#include <vector>

#include <cstddef>

#include "pdsdata/xtc/DetInfo.hh"
namespace PsddlPds {
namespace EvrData {

/** @class PulseConfig

  
*/


class PulseConfig {
public:
  enum {
    Trigger_Shift = 0 /**<  */
  };
  enum {
    Set_Shift = 8 /**<  */
  };
  enum {
    Clear_Shift = 16 /**<  */
  };
  enum {
    Polarity_Shift = 0 /**<  */
  };
  enum {
    Map_Set_Ena_Shift = 1 /**<  */
  };
  enum {
    Map_Reset_Ena_Shift = 2 /**<  */
  };
  enum {
    Map_Trigger_Ena_Shift = 3 /**<  */
  };
  PulseConfig()
  {
  }
  PulseConfig(uint32_t arg__pulse, uint32_t arg__input_control, uint32_t arg__output_control, uint32_t arg__prescale, uint32_t arg__delay, uint32_t arg__width)
    : _pulse(arg__pulse), _input_control(arg__input_control), _output_control(arg__output_control), _prescale(arg__prescale), _delay(arg__delay), _width(arg__width)
  {
  }
  /** internal pulse generation channel */
  uint32_t pulse() const {return _pulse;}
  /** Pulse input control */
  uint32_t _input_control_value() const {return _input_control;}
private:
  int16_t bf_trigger() const {return int16_t(this->_input_control & 0xff);}
  int16_t bf_set() const {return int16_t((this->_input_control>>8) & 0xff);}
  int16_t bf_clear() const {return int16_t((this->_input_control>>16) & 0xff);}
public:
  /** Pulse output control */
  uint32_t _output_control_value() const {return _output_control;}
  uint8_t polarity() const {return uint8_t(this->_output_control & 0x1);}
  uint8_t map_set_enable() const {return uint8_t((this->_output_control>>1) & 0x1);}
  uint8_t map_reset_enable() const {return uint8_t((this->_output_control>>2) & 0x1);}
  uint8_t map_trigger_enable() const {return uint8_t((this->_output_control>>3) & 0x1);}
  /** pulse event prescale */
  uint32_t prescale() const {return _prescale;}
  /** delay in 119MHz clks */
  uint32_t delay() const {return _delay;}
  /** width in 119MHz clks */
  uint32_t width() const {return _width;}
  int16_t trigger() const;
  int16_t set() const;
  int16_t clear() const;
  static uint32_t _sizeof()  {return 24;}
private:
  uint32_t	_pulse;	/**< internal pulse generation channel */
  uint32_t	_input_control;	/**< Pulse input control */
  uint32_t	_output_control;	/**< Pulse output control */
  uint32_t	_prescale;	/**< pulse event prescale */
  uint32_t	_delay;	/**< delay in 119MHz clks */
  uint32_t	_width;	/**< width in 119MHz clks */
};

/** @class PulseConfigV3

  
*/

#pragma pack(push,4)

class PulseConfigV3 {
public:
  PulseConfigV3()
  {
  }
  PulseConfigV3(uint16_t arg__u16PulseId, uint16_t arg__u16Polarity, uint32_t arg__u32Prescale, uint32_t arg__u32Delay, uint32_t arg__u32Width)
    : _u16PulseId(arg__u16PulseId), _u16Polarity(arg__u16Polarity), _u32Prescale(arg__u32Prescale), _u32Delay(arg__u32Delay), _u32Width(arg__u32Width)
  {
  }
  uint16_t pulseId() const {return _u16PulseId;}
  /** 0 -> positive polarity , 1 -> negative polarity */
  uint16_t polarity() const {return _u16Polarity;}
  /** Clock divider */
  uint32_t prescale() const {return _u32Prescale;}
  /** Delay in 119MHz clks */
  uint32_t delay() const {return _u32Delay;}
  /** Width in 119MHz clks */
  uint32_t width() const {return _u32Width;}
  static uint32_t _sizeof()  {return 16;}
private:
  uint16_t	_u16PulseId;
  uint16_t	_u16Polarity;	/**< 0 -> positive polarity , 1 -> negative polarity */
  uint32_t	_u32Prescale;	/**< Clock divider */
  uint32_t	_u32Delay;	/**< Delay in 119MHz clks */
  uint32_t	_u32Width;	/**< Width in 119MHz clks */
};
#pragma pack(pop)

/** @class EventCodeV3

  
*/

#pragma pack(push,4)

class EventCodeV3 {
public:
  EventCodeV3()
  {
  }
  EventCodeV3(uint16_t arg__u16Code, uint16_t arg__u16MaskEventAttr, uint32_t arg__u32MaskTrigger, uint32_t arg__u32MaskSet, uint32_t arg__u32MaskClear)
    : _u16Code(arg__u16Code), _u16MaskEventAttr(arg__u16MaskEventAttr), _u32MaskTrigger(arg__u32MaskTrigger), _u32MaskSet(arg__u32MaskSet), _u32MaskClear(arg__u32MaskClear)
  {
  }
  uint16_t code() const {return _u16Code;}
  uint16_t _u16MaskEventAttr_value() const {return _u16MaskEventAttr;}
  uint8_t isReadout() const {return uint8_t(this->_u16MaskEventAttr & 0x1);}
  uint8_t isTerminator() const {return uint8_t((this->_u16MaskEventAttr>>1) & 0x1);}
  uint32_t maskTrigger() const {return _u32MaskTrigger;}
  uint32_t maskSet() const {return _u32MaskSet;}
  uint32_t maskClear() const {return _u32MaskClear;}
  static uint32_t _sizeof()  {return 16;}
private:
  uint16_t	_u16Code;
  uint16_t	_u16MaskEventAttr;
  uint32_t	_u32MaskTrigger;
  uint32_t	_u32MaskSet;
  uint32_t	_u32MaskClear;
};
#pragma pack(pop)

/** @class EventCodeV4

  
*/

#pragma pack(push,4)

class EventCodeV4 {
public:
  EventCodeV4()
  {
  }
  EventCodeV4(uint16_t arg__u16Code, uint16_t arg__u16MaskEventAttr, uint32_t arg__u32ReportDelay, uint32_t arg__u32ReportWidth, uint32_t arg__u32MaskTrigger, uint32_t arg__u32MaskSet, uint32_t arg__u32MaskClear)
    : _u16Code(arg__u16Code), _u16MaskEventAttr(arg__u16MaskEventAttr), _u32ReportDelay(arg__u32ReportDelay), _u32ReportWidth(arg__u32ReportWidth), _u32MaskTrigger(arg__u32MaskTrigger), _u32MaskSet(arg__u32MaskSet), _u32MaskClear(arg__u32MaskClear)
  {
  }
  uint16_t code() const {return _u16Code;}
  uint16_t _u16MaskEventAttr_value() const {return _u16MaskEventAttr;}
  uint8_t isReadout() const {return uint8_t(this->_u16MaskEventAttr & 0x1);}
  uint8_t isTerminator() const {return uint8_t((this->_u16MaskEventAttr>>1) & 0x1);}
  uint32_t reportDelay() const {return _u32ReportDelay;}
  uint32_t reportWidth() const {return _u32ReportWidth;}
  uint32_t maskTrigger() const {return _u32MaskTrigger;}
  uint32_t maskSet() const {return _u32MaskSet;}
  uint32_t maskClear() const {return _u32MaskClear;}
  static uint32_t _sizeof()  {return 24;}
private:
  uint16_t	_u16Code;
  uint16_t	_u16MaskEventAttr;
  uint32_t	_u32ReportDelay;
  uint32_t	_u32ReportWidth;
  uint32_t	_u32MaskTrigger;
  uint32_t	_u32MaskSet;
  uint32_t	_u32MaskClear;
};
#pragma pack(pop)

/** @class EventCodeV5

  
*/

#pragma pack(push,4)

class EventCodeV5 {
public:
  enum {
    DescSize = 16 /**<  */
  };
  EventCodeV5()
  {
  }
  EventCodeV5(uint16_t arg__u16Code, uint16_t arg__u16MaskEventAttr, uint32_t arg__u32ReportDelay, uint32_t arg__u32ReportWidth, uint32_t arg__u32MaskTrigger, uint32_t arg__u32MaskSet, uint32_t arg__u32MaskClear, const char* arg__desc)
    : _u16Code(arg__u16Code), _u16MaskEventAttr(arg__u16MaskEventAttr), _u32ReportDelay(arg__u32ReportDelay), _u32ReportWidth(arg__u32ReportWidth), _u32MaskTrigger(arg__u32MaskTrigger), _u32MaskSet(arg__u32MaskSet), _u32MaskClear(arg__u32MaskClear)
  {
    std::copy(arg__desc, arg__desc+(16), _desc);
  }
  uint16_t code() const {return _u16Code;}
  uint16_t _u16MaskEventAttr_value() const {return _u16MaskEventAttr;}
  uint8_t isReadout() const {return uint8_t(this->_u16MaskEventAttr & 0x1);}
  uint8_t isTerminator() const {return uint8_t((this->_u16MaskEventAttr>>1) & 0x1);}
  uint8_t isLatch() const {return uint8_t((this->_u16MaskEventAttr>>2) & 0x1);}
  uint32_t reportDelay() const {return _u32ReportDelay;}
  uint32_t reportWidth() const {return _u32ReportWidth;}
  uint32_t maskTrigger() const {return _u32MaskTrigger;}
  uint32_t maskSet() const {return _u32MaskSet;}
  uint32_t maskClear() const {return _u32MaskClear;}
  const char* desc() const {return &_desc[0];}
  static uint32_t _sizeof()  {return 24+(1*(DescSize));}
  /** Method which returns the shape (dimensions) of the data returned by desc() method. */
  std::vector<int> desc_shape() const;
private:
  uint16_t	_u16Code;
  uint16_t	_u16MaskEventAttr;
  uint32_t	_u32ReportDelay;
  uint32_t	_u32ReportWidth;
  uint32_t	_u32MaskTrigger;
  uint32_t	_u32MaskSet;
  uint32_t	_u32MaskClear;
  char	_desc[DescSize];
};
#pragma pack(pop)

/** @class OutputMap

  
*/


class OutputMap {
public:
  enum Source {
    Pulse,
    DBus,
    Prescaler,
    Force_High,
    Force_Low,
  };
  enum Conn {
    FrontPanel,
    UnivIO,
  };
  OutputMap()
  {
  }
  OutputMap(uint32_t arg__v)
    : _v(arg__v)
  {
  }
  uint32_t value() const {return _v;}
  EvrData::OutputMap::Source source() const;
  uint8_t source_id() const;
  EvrData::OutputMap::Conn conn() const;
  uint8_t conn_id() const;
  static uint32_t _sizeof()  {return 4;}
private:
  uint32_t	_v;
};

/** @class ConfigV1

  
*/


class ConfigV1 {
public:
  enum {
    Version = 1 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_EvrConfig /**< XTC type ID value (from Pds::TypeId class) */
  };
  uint32_t npulses() const {return _npulses;}
  uint32_t noutputs() const {return _noutputs;}
  const EvrData::PulseConfig& pulses(uint32_t i0) const {
    ptrdiff_t offset=8;
    const EvrData::PulseConfig* memptr = (const EvrData::PulseConfig*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof();
    return *(const EvrData::PulseConfig*)((const char*)memptr + (i0)*memsize);
  }
  const EvrData::OutputMap& output_maps(uint32_t i0) const {
    ptrdiff_t offset=8+(24*(this->_npulses));
    const EvrData::OutputMap* memptr = (const EvrData::OutputMap*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof();
    return *(const EvrData::OutputMap*)((const char*)memptr + (i0)*memsize);
  }
  uint32_t _sizeof() const {return (8+(EvrData::PulseConfig::_sizeof()*(this->_npulses)))+(EvrData::OutputMap::_sizeof()*(this->_noutputs));}
  /** Method which returns the shape (dimensions) of the data returned by pulses() method. */
  std::vector<int> pulses_shape() const;
  /** Method which returns the shape (dimensions) of the data returned by output_maps() method. */
  std::vector<int> output_maps_shape() const;
private:
  uint32_t	_npulses;
  uint32_t	_noutputs;
  //EvrData::PulseConfig	_pulses[this->_npulses];
  //EvrData::OutputMap	_output_maps[this->_noutputs];
};

/** @class ConfigV2

  
*/


class ConfigV2 {
public:
  enum {
    Version = 2 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_EvrConfig /**< XTC type ID value (from Pds::TypeId class) */
  };
  enum {
    beamOn = 100 /**<  */
  };
  enum {
    baseRate = 40 /**<  */
  };
  enum {
    singleShot = 150 /**<  */
  };
  enum RateCode {
    r120Hz,
    r60Hz,
    r30Hz,
    r10Hz,
    r5Hz,
    r1Hz,
    r0_5Hz,
    Single,
    NumberOfRates,
  };
  enum BeamCode {
    Off,
    On,
  };
  uint32_t opcode() const {return _opcode;}
  uint32_t npulses() const {return _npulses;}
  uint32_t noutputs() const {return _noutputs;}
  const EvrData::PulseConfig& pulses(uint32_t i0) const {
    ptrdiff_t offset=12;
    const EvrData::PulseConfig* memptr = (const EvrData::PulseConfig*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof();
    return *(const EvrData::PulseConfig*)((const char*)memptr + (i0)*memsize);
  }
  const EvrData::OutputMap& output_maps(uint32_t i0) const {
    ptrdiff_t offset=12+(24*(this->_npulses));
    const EvrData::OutputMap* memptr = (const EvrData::OutputMap*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof();
    return *(const EvrData::OutputMap*)((const char*)memptr + (i0)*memsize);
  }
  EvrData::ConfigV2::BeamCode beam() const;
  EvrData::ConfigV2::RateCode rate() const;
  uint32_t _sizeof() const {return (12+(EvrData::PulseConfig::_sizeof()*(this->_npulses)))+(EvrData::OutputMap::_sizeof()*(this->_noutputs));}
  /** Method which returns the shape (dimensions) of the data returned by pulses() method. */
  std::vector<int> pulses_shape() const;
  /** Method which returns the shape (dimensions) of the data returned by output_maps() method. */
  std::vector<int> output_maps_shape() const;
private:
  uint32_t	_opcode;
  uint32_t	_npulses;
  uint32_t	_noutputs;
  //EvrData::PulseConfig	_pulses[this->_npulses];
  //EvrData::OutputMap	_output_maps[this->_noutputs];
};

/** @class ConfigV3

  
*/

#pragma pack(push,4)

class ConfigV3 {
public:
  enum {
    Version = 3 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_EvrConfig /**< XTC type ID value (from Pds::TypeId class) */
  };
  uint32_t neventcodes() const {return _neventcodes;}
  uint32_t npulses() const {return _npulses;}
  uint32_t noutputs() const {return _noutputs;}
  const EvrData::EventCodeV3& eventcodes(uint32_t i0) const {
    ptrdiff_t offset=12;
    const EvrData::EventCodeV3* memptr = (const EvrData::EventCodeV3*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof();
    return *(const EvrData::EventCodeV3*)((const char*)memptr + (i0)*memsize);
  }
  const EvrData::PulseConfigV3& pulses(uint32_t i0) const {
    ptrdiff_t offset=12+(16*(this->_neventcodes));
    const EvrData::PulseConfigV3* memptr = (const EvrData::PulseConfigV3*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof();
    return *(const EvrData::PulseConfigV3*)((const char*)memptr + (i0)*memsize);
  }
  const EvrData::OutputMap& output_maps(uint32_t i0) const {
    ptrdiff_t offset=(12+(16*(this->_neventcodes)))+(16*(this->_npulses));
    const EvrData::OutputMap* memptr = (const EvrData::OutputMap*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof();
    return *(const EvrData::OutputMap*)((const char*)memptr + (i0)*memsize);
  }
  uint32_t _sizeof() const {return ((12+(EvrData::EventCodeV3::_sizeof()*(this->_neventcodes)))+(EvrData::PulseConfigV3::_sizeof()*(this->_npulses)))+(EvrData::OutputMap::_sizeof()*(this->_noutputs));}
  /** Method which returns the shape (dimensions) of the data returned by eventcodes() method. */
  std::vector<int> eventcodes_shape() const;
  /** Method which returns the shape (dimensions) of the data returned by pulses() method. */
  std::vector<int> pulses_shape() const;
  /** Method which returns the shape (dimensions) of the data returned by output_maps() method. */
  std::vector<int> output_maps_shape() const;
private:
  uint32_t	_neventcodes;
  uint32_t	_npulses;
  uint32_t	_noutputs;
  //EvrData::EventCodeV3	_eventcodes[this->_neventcodes];
  //EvrData::PulseConfigV3	_pulses[this->_npulses];
  //EvrData::OutputMap	_output_maps[this->_noutputs];
};
#pragma pack(pop)

/** @class ConfigV4

  
*/

#pragma pack(push,4)

class ConfigV4 {
public:
  enum {
    Version = 4 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_EvrConfig /**< XTC type ID value (from Pds::TypeId class) */
  };
  uint32_t neventcodes() const {return _neventcodes;}
  uint32_t npulses() const {return _npulses;}
  uint32_t noutputs() const {return _noutputs;}
  const EvrData::EventCodeV4& eventcodes(uint32_t i0) const {
    ptrdiff_t offset=12;
    const EvrData::EventCodeV4* memptr = (const EvrData::EventCodeV4*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof();
    return *(const EvrData::EventCodeV4*)((const char*)memptr + (i0)*memsize);
  }
  const EvrData::PulseConfigV3& pulses(uint32_t i0) const {
    ptrdiff_t offset=12+(24*(this->_neventcodes));
    const EvrData::PulseConfigV3* memptr = (const EvrData::PulseConfigV3*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof();
    return *(const EvrData::PulseConfigV3*)((const char*)memptr + (i0)*memsize);
  }
  const EvrData::OutputMap& output_maps(uint32_t i0) const {
    ptrdiff_t offset=(12+(24*(this->_neventcodes)))+(16*(this->_npulses));
    const EvrData::OutputMap* memptr = (const EvrData::OutputMap*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof();
    return *(const EvrData::OutputMap*)((const char*)memptr + (i0)*memsize);
  }
  uint32_t _sizeof() const {return ((12+(EvrData::EventCodeV4::_sizeof()*(this->_neventcodes)))+(EvrData::PulseConfigV3::_sizeof()*(this->_npulses)))+(EvrData::OutputMap::_sizeof()*(this->_noutputs));}
  /** Method which returns the shape (dimensions) of the data returned by eventcodes() method. */
  std::vector<int> eventcodes_shape() const;
  /** Method which returns the shape (dimensions) of the data returned by pulses() method. */
  std::vector<int> pulses_shape() const;
  /** Method which returns the shape (dimensions) of the data returned by output_maps() method. */
  std::vector<int> output_maps_shape() const;
private:
  uint32_t	_neventcodes;
  uint32_t	_npulses;
  uint32_t	_noutputs;
  //EvrData::EventCodeV4	_eventcodes[this->_neventcodes];
  //EvrData::PulseConfigV3	_pulses[this->_npulses];
  //EvrData::OutputMap	_output_maps[this->_noutputs];
};
#pragma pack(pop)

/** @class SequencerEntry

  
*/


class SequencerEntry {
public:
  SequencerEntry()
  {
  }
  SequencerEntry(uint32_t eventcode, uint32_t delay)
    : _value((delay & 0xffffff)|((eventcode & 0xff)<<24))
  {
  }
  uint32_t delay() const {return uint32_t(this->_value & 0xffffff);}
  uint32_t eventcode() const {return uint32_t((this->_value>>24) & 0xff);}
  static uint32_t _sizeof()  {return 4;}
private:
  uint32_t	_value;
};

/** @class SequencerConfigV1

  
*/


class SequencerConfigV1 {
public:
  enum Source {
    r120Hz,
    r60Hz,
    r30Hz,
    r10Hz,
    r5Hz,
    r1Hz,
    r0_5Hz,
    Disable,
  };
  SequencerConfigV1()
  {
  }
  SequencerConfigV1(EvrData::SequencerConfigV1::Source sync_source, EvrData::SequencerConfigV1::Source beam_source, uint32_t length, uint32_t cycles)
    : _source((sync_source & 0xff)|((beam_source & 0xff)<<8)), _length(length), _cycles(cycles)
  {
  }
  EvrData::SequencerConfigV1::Source sync_source() const {return Source(this->_source & 0xff);}
  EvrData::SequencerConfigV1::Source beam_source() const {return Source((this->_source>>8) & 0xff);}
  uint32_t length() const {return _length;}
  uint32_t cycles() const {return _cycles;}
  const EvrData::SequencerEntry& entries(uint32_t i0) const {
    ptrdiff_t offset=12;
    const EvrData::SequencerEntry* memptr = (const EvrData::SequencerEntry*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof();
    return *(const EvrData::SequencerEntry*)((const char*)memptr + (i0)*memsize);
  }
  uint32_t _sizeof() const {return 12+(EvrData::SequencerEntry::_sizeof()*(this->_length));}
  /** Method which returns the shape (dimensions) of the data returned by entries() method. */
  std::vector<int> entries_shape() const;
private:
  uint32_t	_source;
  uint32_t	_length;
  uint32_t	_cycles;
  //EvrData::SequencerEntry	_entries[this->_length];
};

/** @class ConfigV5

  
*/

#pragma pack(push,4)

class ConfigV5 {
public:
  enum {
    Version = 5 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_EvrConfig /**< XTC type ID value (from Pds::TypeId class) */
  };
  uint32_t neventcodes() const {return _neventcodes;}
  uint32_t npulses() const {return _npulses;}
  uint32_t noutputs() const {return _noutputs;}
  const EvrData::EventCodeV5& eventcodes(uint32_t i0) const {
    ptrdiff_t offset=12;
    const EvrData::EventCodeV5* memptr = (const EvrData::EventCodeV5*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof();
    return *(const EvrData::EventCodeV5*)((const char*)memptr + (i0)*memsize);
  }
  const EvrData::PulseConfigV3& pulses(uint32_t i0) const {
    ptrdiff_t offset=12+(40*(this->_neventcodes));
    const EvrData::PulseConfigV3* memptr = (const EvrData::PulseConfigV3*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof();
    return *(const EvrData::PulseConfigV3*)((const char*)memptr + (i0)*memsize);
  }
  const EvrData::OutputMap& output_maps(uint32_t i0) const {
    ptrdiff_t offset=(12+(40*(this->_neventcodes)))+(16*(this->_npulses));
    const EvrData::OutputMap* memptr = (const EvrData::OutputMap*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof();
    return *(const EvrData::OutputMap*)((const char*)memptr + (i0)*memsize);
  }
  const EvrData::SequencerConfigV1& seq_config() const {
    ptrdiff_t offset=((12+(40*(this->_neventcodes)))+(16*(this->_npulses)))+(4*(this->_noutputs));
    const EvrData::SequencerConfigV1* memptr = (const EvrData::SequencerConfigV1*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof();
    return *(const EvrData::SequencerConfigV1*)((const char*)memptr + (0)*memsize);
  }
  /** Method which returns the shape (dimensions) of the data returned by eventcodes() method. */
  std::vector<int> eventcodes_shape() const;
  /** Method which returns the shape (dimensions) of the data returned by pulses() method. */
  std::vector<int> pulses_shape() const;
  /** Method which returns the shape (dimensions) of the data returned by output_maps() method. */
  std::vector<int> output_maps_shape() const;
private:
  uint32_t	_neventcodes;
  uint32_t	_npulses;
  uint32_t	_noutputs;
  //EvrData::EventCodeV5	_eventcodes[this->_neventcodes];
  //EvrData::PulseConfigV3	_pulses[this->_npulses];
  //EvrData::OutputMap	_output_maps[this->_noutputs];
  //EvrData::SequencerConfigV1	_seq_config;
};
#pragma pack(pop)

/** @class FIFOEvent

  
*/


class FIFOEvent {
public:
  FIFOEvent()
  {
  }
  FIFOEvent(uint32_t arg__timestampHigh, uint32_t arg__timestampLow, uint32_t arg__eventCode)
    : _timestampHigh(arg__timestampHigh), _timestampLow(arg__timestampLow), _eventCode(arg__eventCode)
  {
  }
  uint32_t timestampHigh() const {return _timestampHigh;}
  uint32_t timestampLow() const {return _timestampLow;}
  uint32_t eventCode() const {return _eventCode;}
  static uint32_t _sizeof()  {return 12;}
private:
  uint32_t	_timestampHigh;
  uint32_t	_timestampLow;
  uint32_t	_eventCode;
};

/** @class DataV3

  
*/


class DataV3 {
public:
  enum {
    Version = 3 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_EvrData /**< XTC type ID value (from Pds::TypeId class) */
  };
  uint32_t numFifoEvents() const {return _u32NumFifoEvents;}
  const EvrData::FIFOEvent& fifoEvents(uint32_t i0) const {
    ptrdiff_t offset=4;
    const EvrData::FIFOEvent* memptr = (const EvrData::FIFOEvent*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof();
    return *(const EvrData::FIFOEvent*)((const char*)memptr + (i0)*memsize);
  }
  uint32_t _sizeof() const {return 4+(EvrData::FIFOEvent::_sizeof()*(this->_u32NumFifoEvents));}
  /** Method which returns the shape (dimensions) of the data returned by fifoEvents() method. */
  std::vector<int> fifoEvents_shape() const;
private:
  uint32_t	_u32NumFifoEvents;
  //EvrData::FIFOEvent	_fifoEvents[this->_u32NumFifoEvents];
};

/** @class IOChannel

  
*/


class IOChannel {
public:
  enum {
    NameLength = 12 /**<  */
  };
  enum {
    MaxInfos = 8 /**<  */
  };
  const char* name() const {return &_name[0];}
  uint32_t ninfo() const {return _ninfo;}
  const Pds::DetInfo& infos(uint32_t i0) const {return _info[i0];}
  static uint32_t _sizeof()  {return ((0+(1*(NameLength)))+4)+(8*(MaxInfos));}
  /** Method which returns the shape (dimensions) of the data returned by name() method. */
  std::vector<int> name_shape() const;
  /** Method which returns the shape (dimensions) of the data returned by infos() method. */
  std::vector<int> infos_shape() const;
private:
  char	_name[NameLength];
  uint32_t	_ninfo;
  Pds::DetInfo	_info[MaxInfos];
};

/** @class IOConfigV1

  
*/

#pragma pack(push,4)

class IOConfigV1 {
public:
  enum {
    Version = 1 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_EvrIOConfig /**< XTC type ID value (from Pds::TypeId class) */
  };
  uint16_t nchannels() const {return _nchannels;}
  const EvrData::IOChannel& channels(uint32_t i0) const {
    ptrdiff_t offset=4;
    const EvrData::IOChannel* memptr = (const EvrData::IOChannel*)(((const char*)this)+offset);
    size_t memsize = memptr->_sizeof();
    return *(const EvrData::IOChannel*)((const char*)memptr + (i0)*memsize);
  }
  EvrData::OutputMap::Conn conn() const;
  uint32_t _sizeof() const {return 4+(EvrData::IOChannel::_sizeof()*(this->_nchannels));}
  /** Method which returns the shape (dimensions) of the data returned by channels() method. */
  std::vector<int> channels_shape() const;
private:
  uint16_t	_conn;
  uint16_t	_nchannels;
  //EvrData::IOChannel	_channels[this->_nchannels];
};
#pragma pack(pop)
} // namespace EvrData
} // namespace PsddlPds
#endif // PSDDLPDS_EVR_DDL_H
