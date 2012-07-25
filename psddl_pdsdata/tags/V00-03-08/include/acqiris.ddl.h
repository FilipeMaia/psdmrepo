#ifndef PSDDLPDS_ACQIRIS_DDL_H
#define PSDDLPDS_ACQIRIS_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include <vector>
#include <cstddef>
#include "pdsdata/xtc/TypeId.hh"
#include "ndarray/ndarray.h"
namespace PsddlPds {
namespace Acqiris {

/** @class VertV1

  Class containing Acqiris configuration data for vertical axis.
*/

#pragma pack(push,4)

class VertV1 {
public:
  enum { Version = 1 /**< XTC type version number */ };

  /** Coupling modes */
  enum Coupling {
    GND,
    DC,
    AC,
    DC50ohm,
    AC50ohm,
  };
  enum Bandwidth {
    None,
    MHz25,
    MHz700,
    MHz200,
    MHz20,
    MHz35,
  };
  VertV1()
  {
  }
  VertV1(double arg__fullScale, double arg__offset, uint32_t arg__coupling, uint32_t arg__bandwidth)
    : _fullScale(arg__fullScale), _offset(arg__offset), _coupling(arg__coupling), _bandwidth(arg__bandwidth)
  {
  }
  /** Full vertical scale. */
  double fullScale() const { return _fullScale; }
  /** Offset value. */
  double offset() const { return _offset; }
  /** Coupling mode. */
  uint32_t coupling() const { return _coupling; }
  /** Bandwidth enumeration. */
  uint32_t bandwidth() const { return _bandwidth; }
  /** Calculated slope. */
  double slope() const;
  static uint32_t _sizeof()  { return 24; }
private:
  double	_fullScale;	/**< Full vertical scale. */
  double	_offset;	/**< Offset value. */
  uint32_t	_coupling;	/**< Coupling mode. */
  uint32_t	_bandwidth;	/**< Bandwidth enumeration. */
};
#pragma pack(pop)

/** @class HorizV1

  Class containing Acqiris configuration data for horizontal axis.
*/

#pragma pack(push,4)

class HorizV1 {
public:
  enum { Version = 1 /**< XTC type version number */ };
  HorizV1()
  {
  }
  HorizV1(double arg__sampInterval, double arg__delayTime, uint32_t arg__nbrSamples, uint32_t arg__nbrSegments)
    : _sampInterval(arg__sampInterval), _delayTime(arg__delayTime), _nbrSamples(arg__nbrSamples), _nbrSegments(arg__nbrSegments)
  {
  }
  /** Interval for single sample. */
  double sampInterval() const { return _sampInterval; }
  /** Delay time. */
  double delayTime() const { return _delayTime; }
  /** Number of samples. */
  uint32_t nbrSamples() const { return _nbrSamples; }
  /** Number of segments. */
  uint32_t nbrSegments() const { return _nbrSegments; }
  static uint32_t _sizeof()  { return 24; }
private:
  double	_sampInterval;	/**< Interval for single sample. */
  double	_delayTime;	/**< Delay time. */
  uint32_t	_nbrSamples;	/**< Number of samples. */
  uint32_t	_nbrSegments;	/**< Number of segments. */
};
#pragma pack(pop)

/** @class TrigV1

  Class containing Acqiris configuration data for triggering.
*/

#pragma pack(push,4)

class TrigV1 {
public:
  enum { Version = 1 /**< XTC type version number */ };

  /** Trigger source. */
  enum Source {
    Internal = 1,
    External = -1,
  };
  enum Coupling {
    DC = 0,
    AC = 1,
    HFreject = 2,
    DC50ohm = 3,
    AC50ohm = 4,
  };

  /** Triggering slope. */
  enum Slope {
    Positive,
    Negative,
    OutOfWindow,
    IntoWindow,
    HFDivide,
    SpikeStretcher,
  };
  TrigV1()
  {
  }
  TrigV1(uint32_t arg__coupling, uint32_t arg__input, uint32_t arg__slope, double arg__level)
    : _coupling(arg__coupling), _input(arg__input), _slope(arg__slope), _level(arg__level)
  {
  }
  uint32_t coupling() const { return _coupling; }
  /** Trigger source */
  uint32_t input() const { return _input; }
  /** Triggering slope. */
  uint32_t slope() const { return _slope; }
  /** Trigger level. */
  double level() const { return _level; }
  static uint32_t _sizeof()  { return 20; }
private:
  uint32_t	_coupling;
  uint32_t	_input;	/**< Trigger source */
  uint32_t	_slope;	/**< Triggering slope. */
  double	_level;	/**< Trigger level. */
};
#pragma pack(pop)

/** @class ConfigV1

  Class containing all Acqiris configuration data.
*/

#pragma pack(push,4)

class ConfigV1 {
public:
  enum { TypeId = Pds::TypeId::Id_AcqConfig /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  enum { MaxChan = 20 /**< Maximum number of the configured channels. */ };
  /** Number of ADCs per channel. */
  uint32_t nbrConvertersPerChannel() const { return _nbrConvertersPerChannel; }
  /** Bit mask for channels. */
  uint32_t channelMask() const { return _channelMask; }
  /** Total number of banks. */
  uint32_t nbrBanks() const { return _nbrBanks; }
  /** Trigger configuration. */
  const Acqiris::TrigV1& trig() const { return _trig; }
  /** Configuration for horizontal axis */
  const Acqiris::HorizV1& horiz() const { return _horiz; }
  /** Configuration for vertical axis (one per channel). */
  ndarray<Acqiris::VertV1, 1> vert() const { return make_ndarray(&_vert[0], MaxChan); }
  /** Number of channels calculated from channel bit mask. */
  uint32_t nbrChannels() const;
  static uint32_t _sizeof()  { return ((12+(Acqiris::TrigV1::_sizeof()))+(Acqiris::HorizV1::_sizeof()))+(Acqiris::VertV1::_sizeof()*(MaxChan)); }
private:
  uint32_t	_nbrConvertersPerChannel;	/**< Number of ADCs per channel. */
  uint32_t	_channelMask;	/**< Bit mask for channels. */
  uint32_t	_nbrBanks;	/**< Total number of banks. */
  Acqiris::TrigV1	_trig;	/**< Trigger configuration. */
  Acqiris::HorizV1	_horiz;	/**< Configuration for horizontal axis */
  Acqiris::VertV1	_vert[MaxChan];	/**< Configuration for vertical axis (one per channel). */
};
#pragma pack(pop)

/** @class TimestampV1

  Class representing Acqiris timestamp value.
*/

#pragma pack(push,4)

class TimestampV1 {
public:
  enum { Version = 1 /**< XTC type version number */ };
  TimestampV1()
  {
  }
  TimestampV1(double arg__horPos, uint32_t arg__timeStampLo, uint32_t arg__timeStampHi)
    : _horPos(arg__horPos), _timeStampLo(arg__timeStampLo), _timeStampHi(arg__timeStampHi)
  {
  }
  double pos() const { return _horPos; }
  uint32_t timeStampLo() const { return _timeStampLo; }
  uint32_t timeStampHi() const { return _timeStampHi; }
  /** Full timestamp as 64-bit number. */
  uint64_t value() const;
  static uint32_t _sizeof()  { return 16; }
private:
  double	_horPos;
  uint32_t	_timeStampLo;
  uint32_t	_timeStampHi;
};
#pragma pack(pop)

/** @class DataDescV1Elem

  Class representing Acqiris waveforms from single channel.
*/

class ConfigV1;
#pragma pack(push,4)

class DataDescV1Elem {
public:
  enum { Version = 1 /**< XTC type version number */ };
  enum { NumberOfBits = 10 };
  enum { BitShift = 6 };
  enum { _extraSize = 32 };
  /** Number of samples in one segment. */
  uint32_t nbrSamplesInSeg() const { return _returnedSamplesPerSeg; }
  uint32_t indexFirstPoint() const { return _indexFirstPoint; }
  /** Number of segments. */
  uint32_t nbrSegments() const { return _returnedSegments; }
  /** Timestamps, one timestamp per segment. */
  ndarray<Acqiris::TimestampV1, 1> timestamp(const Acqiris::ConfigV1& cfg) const { ptrdiff_t offset=64;
  Acqiris::TimestampV1* data = (Acqiris::TimestampV1*)(((const char*)this)+offset);
  return make_ndarray(data, cfg.horiz().nbrSegments()); }
  /** Waveforms data, two-dimensional array [nbrSegments()]*[nbrSamplesInSeg()] */
  ndarray<int16_t, 2> waveforms(const Acqiris::ConfigV1& cfg) const { ptrdiff_t offset=(64+(16*(cfg.horiz().nbrSegments())))+(2*(this->indexFirstPoint()));
  int16_t* data = (int16_t*)(((const char*)this)+offset);
  return make_ndarray(data, cfg.horiz().nbrSegments(), cfg.horiz().nbrSamples()); }
  uint32_t _sizeof(const Acqiris::ConfigV1& cfg) const { return (((64+(Acqiris::TimestampV1::_sizeof()*(cfg.horiz().nbrSegments())))+(2*(this->indexFirstPoint())))+(2*(cfg.horiz().nbrSegments())*(cfg.horiz().nbrSamples())))+(2*(_extraSize-this->indexFirstPoint())); }
private:
  uint32_t	_returnedSamplesPerSeg;	/**< Number of samples in one segment. */
  uint32_t	_indexFirstPoint;
  double	_sampTime;
  double	_vGain;
  double	_vOffset;
  uint32_t	_returnedSegments;	/**< Number of segments. */
  uint32_t	_nbrAvgWforms;
  uint32_t	_actualTriggersInAcqLo;
  uint32_t	_actualTriggersInAcqHi;
  uint32_t	_actualDataSize;
  uint32_t	_reserved2;
  double	_reserved3;
  //Acqiris::TimestampV1	_timestamps[cfg.horiz().nbrSegments()];
  //int16_t	_skip[this->indexFirstPoint()];
  //int16_t	_waveforms[cfg.horiz().nbrSegments()][cfg.horiz().nbrSamples()];
  //int16_t	_extraSpace[_extraSize-this->indexFirstPoint()];
};
#pragma pack(pop)

/** @class DataDescV1

  Class containing waveform data (DataDescV1Elem) for all channels.
*/

class ConfigV1;
#pragma pack(push,4)

class DataDescV1 {
public:
  enum { TypeId = Pds::TypeId::Id_AcqWaveform /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  /** Waveform data, one object per channel. */
  const Acqiris::DataDescV1Elem& data(const Acqiris::ConfigV1& cfg, uint32_t i0) const { const char* memptr = ((const char*)this)+0;
  for (uint32_t i=0; i != i0; ++ i) {
    memptr += ((const Acqiris::DataDescV1Elem*)memptr)->_sizeof(cfg);
  }
  return *(const Acqiris::DataDescV1Elem*)(memptr); }
  /** Method which returns the shape (dimensions) of the data returned by data() method. */
  std::vector<int> data_shape(const Acqiris::ConfigV1& cfg) const;
private:
  //Acqiris::DataDescV1Elem	_data[cfg.nbrChannels()];
};
#pragma pack(pop)

/** @class TdcChannel

  Configuration for Acqiris TDC channel.
*/

#pragma pack(push,4)

class TdcChannel {
public:

  /** Types of channels. */
  enum Channel {
    Veto = -2,
    Common = -1,
    Input1 = 1,
    Input2 = 2,
    Input3 = 3,
    Input4 = 4,
    Input5 = 5,
    Input6 = 6,
  };
  enum Mode {
    Active = 0,
    Inactive = 1,
  };
  enum Slope {
    Positive,
    Negative,
  };
  TdcChannel()
  {
  }
  TdcChannel(uint32_t arg__channel, Acqiris::TdcChannel::Slope arg_bf__bf_slope, Acqiris::TdcChannel::Mode arg_bf__bf_mode, double arg__level)
    : _channel(arg__channel), _mode((arg_bf__bf_slope & 0x1)|((arg_bf__bf_mode & 0x1)<<31)), _level(arg__level)
  {
  }
  /** Channel type as integer number, clients should use channel() method instead. */
  uint32_t _channel_int() const { return _channel; }
  /** Bitfield value, should not be used directly. Use mode() and slope()
                in the client code. */
  uint32_t _mode_int() const { return _mode; }
  Acqiris::TdcChannel::Slope slope() const { return Slope(this->_mode & 0x1); }
  Acqiris::TdcChannel::Mode mode() const { return Mode((this->_mode>>31) & 0x1); }
  double level() const { return _level; }
  Acqiris::TdcChannel::Channel channel() const { return Channel(this->_channel); }
  static uint32_t _sizeof()  { return 16; }
private:
  uint32_t	_channel;	/**< Channel type as integer number, clients should use channel() method instead. */
  uint32_t	_mode;	/**< Bitfield value, should not be used directly. Use mode() and slope()
                in the client code. */
  double	_level;
};
#pragma pack(pop)

/** @class TdcAuxIO

  configuration for auxiliary IO channel.
*/

#pragma pack(push,4)

class TdcAuxIO {
public:
  enum Channel {
    IOAux1 = 1,
    IOAux2 = 2,
  };
  enum Mode {
    BankSwitch = 1,
    Marker = 2,
    OutputLo = 32,
    OutputHi = 33,
  };
  enum Termination {
    ZHigh = 0,
    Z50 = 1,
  };
  TdcAuxIO()
  {
  }
  TdcAuxIO(uint32_t arg__channel, uint32_t arg__signal, uint32_t arg__qualifier)
    : _channel(arg__channel), _signal(arg__signal), _qualifier(arg__qualifier)
  {
  }
  /** Channel type as integer number, clients should use channel() method instead. */
  uint32_t channel_int() const { return _channel; }
  /** Mode as integer number, clients should use mode() method instead. */
  uint32_t signal_int() const { return _signal; }
  /** Termination as integer number, clients should use term() method instead. */
  uint32_t qualifier_int() const { return _qualifier; }
  Acqiris::TdcAuxIO::Channel channel() const { return Channel(this->_channel); }
  Acqiris::TdcAuxIO::Mode mode() const { return Mode(this->_signal); }
  Acqiris::TdcAuxIO::Termination term() const { return Termination(this->_qualifier); }
  static uint32_t _sizeof()  { return 12; }
private:
  uint32_t	_channel;	/**< Channel type as integer number, clients should use channel() method instead. */
  uint32_t	_signal;	/**< Mode as integer number, clients should use mode() method instead. */
  uint32_t	_qualifier;	/**< Termination as integer number, clients should use term() method instead. */
};
#pragma pack(pop)

/** @class TdcVetoIO

  Class with configuration data for Veto IO channel.
*/

#pragma pack(push,4)

class TdcVetoIO {
public:
  enum Channel {
    ChVeto = 13,
  };
  enum Mode {
    Veto = 1,
    SwitchVeto = 2,
    InvertedVeto = 3,
    InvertedSwitchVeto = 4,
  };
  enum Termination {
    ZHigh = 0,
    Z50 = 1,
  };
  TdcVetoIO()
  {
  }
  TdcVetoIO(uint32_t mode, uint32_t term)
    : _channel(ChVeto), _signal(mode), _qualifier(term)
  {
  }
  /** Mode as integer number, clients should use mode() method instead. */
  uint32_t signal_int() const { return _signal; }
  /** Termination as integer number, clients should use term() method instead. */
  uint32_t qualifier_int() const { return _qualifier; }
  Acqiris::TdcVetoIO::Channel channel() const { return Channel(this->_channel); }
  Acqiris::TdcVetoIO::Mode mode() const { return Mode(this->_signal); }
  Acqiris::TdcVetoIO::Termination term() const { return Termination(this->_qualifier); }
  static uint32_t _sizeof()  { return 12; }
private:
  uint32_t	_channel;	/**< Channel type as integer number, clients should use channel() method instead. */
  uint32_t	_signal;	/**< Mode as integer number, clients should use mode() method instead. */
  uint32_t	_qualifier;	/**< Termination as integer number, clients should use term() method instead. */
};
#pragma pack(pop)

/** @class TdcConfigV1

  Class with complete Acqiris TDC configuration.
*/

#pragma pack(push,4)

class TdcConfigV1 {
public:
  enum { TypeId = Pds::TypeId::Id_AcqTdcConfig /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  enum { NChannels = 8 /**< Total number of channel configurations. */ };
  enum { NAuxIO = 2 /**< Total number of auxiliary IO configurations. */ };
  /** Channel configurations, one object per channel. */
  ndarray<Acqiris::TdcChannel, 1> channels() const { return make_ndarray(&_channel[0], NChannels); }
  /** Axiliary configurations, one object per channel. */
  ndarray<Acqiris::TdcAuxIO, 1> auxio() const { return make_ndarray(&_auxIO[0], NAuxIO); }
  const Acqiris::TdcVetoIO& veto() const { return _veto; }
  static uint32_t _sizeof()  { return ((0+(Acqiris::TdcChannel::_sizeof()*(NChannels)))+(Acqiris::TdcAuxIO::_sizeof()*(NAuxIO)))+(Acqiris::TdcVetoIO::_sizeof()); }
private:
  Acqiris::TdcChannel	_channel[NChannels];	/**< Channel configurations, one object per channel. */
  Acqiris::TdcAuxIO	_auxIO[NAuxIO];	/**< Axiliary configurations, one object per channel. */
  Acqiris::TdcVetoIO	_veto;
};
#pragma pack(pop)

/** @class TdcDataV1_Item

  Base class for all Acqiris TDC data objects.
*/


class TdcDataV1_Item {
public:

  /** Enum for types of data objects. Comm means data object has TdcDataV1Common
	          type, AuxIO means TdcDataV1Marker class, all others are for TdcDataV1Channel. */
  enum Source {
    Comm,
    Chan1,
    Chan2,
    Chan3,
    Chan4,
    Chan5,
    Chan6,
    AuxIO,
  };
  TdcDataV1_Item()
  {
  }
  TdcDataV1_Item(uint32_t arg_bf__bf_val, Acqiris::TdcDataV1_Item::Source arg_bf__bf_source, uint8_t arg_bf__bf_ovf)
    : _value((arg_bf__bf_val & 0xfffffff)|((arg_bf__bf_source & 0x7)<<28)|((arg_bf__bf_ovf & 0x1)<<31))
  {
  }
  /** Value as integer number whiis composed of several bit fields. Do not use value directly,
                instead cast this object to one of the actual types and use corresponding methods. */
  uint32_t value() const { return _value; }
  uint32_t bf_val_() const { return uint32_t(this->_value & 0xfffffff); }
  /** Source of this data object, use returned enum to distinguish between actual 
                types of data objecs and cast appropriately. */
  Acqiris::TdcDataV1_Item::Source source() const { return Source((this->_value>>28) & 0x7); }
  uint8_t bf_ofv_() const { return uint8_t((this->_value>>31) & 0x1); }
  static uint32_t _sizeof()  { return 4; }
private:
  uint32_t	_value;	/**< Value as integer number whiis composed of several bit fields. Do not use value directly,
                instead cast this object to one of the actual types and use corresponding methods. */
};

/** @class TdcDataV1Common

  Class for the "common" TDC data object.
*/


class TdcDataV1Common: public TdcDataV1_Item {
public:
  /** Returns number of hits. */
  uint32_t nhits() const;
  /** Returns overflow status. */
  uint8_t overflow() const;
  static uint32_t _sizeof()  { return Acqiris::TdcDataV1_Item::_sizeof(); }
};

/** @class TdcDataV1Channel

  Class for the "channel" TDC data object.
*/


class TdcDataV1Channel: public TdcDataV1_Item {
public:
  /** Returns number of ticks. */
  uint32_t ticks() const;
  /** Returns overflow status. */
  uint8_t overflow() const;
  /** Ticks converted to time, tick resolution is 50 picosecond. */
  double time() const;
  static uint32_t _sizeof()  { return Acqiris::TdcDataV1_Item::_sizeof(); }
};

/** @class TdcDataV1Marker

  Class for the "marker" TDC data object.
*/


class TdcDataV1Marker: public TdcDataV1_Item {
public:

  /** Enum for the type of marker. */
  enum Type {
    AuxIOSwitch = 0,
    EventCntSwitch = 1,
    MemFullSwitch = 2,
    AuxIOMarker = 16,
  };
  /** Returns type of the marker. */
  Acqiris::TdcDataV1Marker::Type type() const;
  static uint32_t _sizeof()  { return Acqiris::TdcDataV1_Item::_sizeof(); }
};

/** @class TdcDataV1

  Acqiris TDS data object is a container for TdcDataV1_Item object (or their
            sub-types).
*/


class TdcDataV1 {
public:
  enum { TypeId = Pds::TypeId::Id_AcqTdcData /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  /** Access TDC data items. The data_shape() method should be used to 
            obtain the number of elements. */
  ndarray<Acqiris::TdcDataV1_Item, 1> data() const { ptrdiff_t offset=0;
  Acqiris::TdcDataV1_Item* data = (Acqiris::TdcDataV1_Item*)(((const char*)this)+offset);
  return make_ndarray(data, 0); }
  static uint32_t _sizeof()  { return ~uint32_t(0); }
private:
  //Acqiris::TdcDataV1_Item	_data[None];
};
} // namespace Acqiris
} // namespace PsddlPds
#endif // PSDDLPDS_ACQIRIS_DDL_H
