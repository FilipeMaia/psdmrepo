@package Acqiris  {


//------------------ VertV1 ------------------
/* Class containing Acqiris configuration data for vertical axis. */
@type VertV1
  [[value_type]]
  [[pack(4)]]
{
  /* Coupling modes */
  @enum Coupling (uint32_t) {
    GND,
    DC,
    AC,
    DC50ohm,
    AC50ohm,
  }
  @enum Bandwidth (uint32_t) {
    None,
    MHz25,
    MHz700,
    MHz200,
    MHz20,
    MHz35,
  }

  double _fullScale -> fullScale;	/* Full vertical scale. */
  double _offset -> offset;	/* Offset value. */
  uint32_t _coupling -> coupling;	/* Coupling mode. */
  uint32_t _bandwidth -> bandwidth;	/* Bandwidth enumeration. */

  /* Calculated slope. */
  double slope()
  [[language("C++")]] @{ 
    return @self.fullScale() / ((1 << Acqiris::DataDescV1Elem::NumberOfBits)*(1 << Acqiris::DataDescV1Elem::BitShift)); 
  @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ HorizV1 ------------------
/* Class containing Acqiris configuration data for horizontal axis. */
@type HorizV1
  [[value_type]]
  [[pack(4)]]
{
  double _sampInterval -> sampInterval;	/* Interval for single sample. */
  double _delayTime -> delayTime;	/* Delay time. */
  uint32_t _nbrSamples -> nbrSamples;	/* Number of samples. */
  uint32_t _nbrSegments -> nbrSegments;	/* Number of segments. */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ TrigV1 ------------------
/* Class containing Acqiris configuration data for triggering. */
@type TrigV1
  [[value_type]]
  [[pack(4)]]
{
  /* Trigger source. */
  @enum Source (int32_t) {
    Internal = 1,
    External = -1,
  }
  @enum Coupling (uint32_t) {
    DC = 0,
    AC = 1,
    HFreject = 2,
    DC50ohm = 3,
    AC50ohm = 4,
  }
  /* Triggering slope. */
  @enum Slope (uint32_t) {
    Positive,
    Negative,
    OutOfWindow,
    IntoWindow,
    HFDivide,
    SpikeStretcher,
  }

  uint32_t _coupling -> coupling;
  uint32_t _input -> input;	/* Trigger source */
  uint32_t _slope -> slope;	/* Triggering slope. */
  double _level -> level;	/* Trigger level. */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ ConfigV1 ------------------
/* Class containing all Acqiris configuration data. */
@type ConfigV1
  [[type_id(Id_AcqConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  /* Maximum number of the configured channels. */
  @const int32_t MaxChan = 20;

  uint32_t _nbrConvertersPerChannel -> nbrConvertersPerChannel;	/* Number of ADCs per channel. */
  uint32_t _channelMask -> channelMask;	/* Bit mask for channels. */
  uint32_t _nbrBanks -> nbrBanks;	/* Total number of banks. */
  TrigV1 _trig -> trig;	/* Trigger configuration. */
  HorizV1 _horiz -> horiz;	/* Configuration for horizontal axis */
  VertV1 _vert[MaxChan] -> vert  [[shape_method(vert_shape)]];	/* Configuration for vertical axis (one per channel). */

  /* Number of channels calculated from channel bit mask. */
  uint32_t nbrChannels()
  [[language("C++")]] @{ return __builtin_popcount(@self._channelMask); @}

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ TimestampV1 ------------------
/* Class representing Acqiris timestamp value. */
@type TimestampV1
  [[value_type]]
  [[pack(4)]]
{
  double _horPos -> pos;	/* Horizontal position, for the segment, of the first (nominal) data point with respect 
            to the origin of the nominal trigger delay in seconds. */
  uint32_t _timeStampLo -> timeStampLo;
  uint32_t _timeStampHi -> timeStampHi;

  /* 64-bit trigger timestamp, in units of picoseconds. The timestamp is the trigger time 
                with respect to an arbitrary time origin. */
  uint64_t value()
  [[language("C++")]] @{ return (((uint64_t)@self._timeStampHi)<<32) + @self._timeStampLo; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ DataDescV1Elem ------------------
/* Class representing Acqiris waveforms from single channel. */
@type DataDescV1Elem
  [[pack(4)]]
  [[config(ConfigV1)]]
{
  @const int32_t NumberOfBits = 10;
  @const int32_t BitShift = 6;
  @const int32_t _extraSize = 32;

  uint32_t _returnedSamplesPerSeg -> nbrSamplesInSeg;	/* Number of samples in one segment. */
  uint32_t _indexFirstPoint -> indexFirstPoint;
  double _sampTime;
  double _vGain;
  double _vOffset;
  uint32_t _returnedSegments -> nbrSegments;	/* Number of segments. */
  uint32_t _nbrAvgWforms;
  uint32_t _actualTriggersInAcqLo;
  uint32_t _actualTriggersInAcqHi;
  uint32_t _actualDataSize;
  uint32_t _reserved2;
  double _reserved3;
  TimestampV1 _timestamps[@config.horiz().nbrSegments()] -> timestamp  [[shape_method(timestamps_shape)]];	/* Timestamps, one timestamp per segment. */
  int16_t _skip[@self.indexFirstPoint()];
  int16_t _waveforms[@config.horiz().nbrSegments()][@config.horiz().nbrSamples()] -> waveforms  [[shape_method(waveforms_shape)]];	/* Waveforms data, two-dimensional array [nbrSegments()]*[nbrSamplesInSeg()]. Note that 
            unlike in pdsdata this already takes into account value of the indexFirstPoint so
            that client code does not need to correct for this offset. */
  int16_t _extraSpace[_extraSize-@self.indexFirstPoint()];

  /* Contructor for valid sizeof estimate */
  @init(indexFirstPoint -> _indexFirstPoint) [[inline]];
}


//------------------ DataDescV1 ------------------
/* Class containing waveform data (DataDescV1Elem) for all channels. */
@type DataDescV1
  [[type_id(Id_AcqWaveform, 1)]]
  [[no_sizeof]]
  [[pack(4)]]
  [[config(ConfigV1)]]
{
  DataDescV1Elem _data[@config.nbrChannels()] -> data  [[shape_method(data_shape)]];	/* Waveform data, one object per channel. */
}


//------------------ TdcChannel ------------------
/* Configuration for Acqiris TDC channel. */
@type TdcChannel
  [[value_type]]
  [[pack(4)]]
{
  /* Types of channels. */
  @enum Channel (int32_t) {
    Veto = -2,
    Common = -1,
    Input1 = 1,
    Input2 = 2,
    Input3 = 3,
    Input4 = 4,
    Input5 = 5,
    Input6 = 6,
  }
  @enum Mode (uint16_t) {
    Active = 0,
    Inactive = 1,
  }
  @enum Slope (uint16_t) {
    Positive,
    Negative,
  }

  Channel _channel -> channel;	/* Channel type as integer number, clients should use channel() method instead. */
  uint32_t _mode -> _mode_int {	/* Bitfield value, should not be used directly. Use mode() and slope()
                in the client code. */
    Slope _bf_slope:1 -> slope;
    uint32_t _bf_pad:30;
    Mode _bf_mode:1 -> mode;
  }
  double _level -> level;

  /* Standard constructor */
  @init()  [[auto, inline]];

}


//------------------ TdcAuxIO ------------------
/* configuration for auxiliary IO channel. */
@type TdcAuxIO
  [[value_type]]
  [[pack(4)]]
{
  @enum Channel (uint32_t) {
    IOAux1 = 1,
    IOAux2 = 2,
  }
  @enum Mode (uint32_t) {
    BankSwitch = 1,
    Marker = 2,
    OutputLo = 32,
    OutputHi = 33,
  }
  @enum Termination (uint32_t) {
    ZHigh = 0,
    Z50 = 1,
  }

  Channel _channel -> channel;
  Mode _signal -> mode;
  Termination _qualifier -> term;

  /* Standard constructor */
  @init()  [[auto, inline]];

}


//------------------ TdcVetoIO ------------------
/* Class with configuration data for Veto IO channel. */
@type TdcVetoIO
  [[value_type]]
  [[pack(4)]]
{
  @enum Channel (uint32_t) {
    ChVeto = 13,
  }
  @enum Mode (uint32_t) {
    Veto = 1,
    SwitchVeto = 2,
    InvertedVeto = 3,
    InvertedSwitchVeto = 4,
  }
  @enum Termination (uint32_t) {
    ZHigh = 0,
    Z50 = 1,
  }

  Channel _channel -> channel;
  Mode _signal -> mode;
  Termination _qualifier -> term;

  /* Standard constructor */
  @init(mode -> _signal, term -> _qualifier)
    _channel(ChVeto)  [[inline]];

}


//------------------ TdcConfigV1 ------------------
/* Class with complete Acqiris TDC configuration. */
@type TdcConfigV1
  [[type_id(Id_AcqTdcConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  /* Total number of channel configurations. */
  @const int32_t NChannels = 8;
  /* Total number of auxiliary IO configurations. */
  @const int32_t NAuxIO = 2;

  TdcChannel _channel[NChannels] -> channels  [[shape_method(channels_shape)]];	/* Channel configurations, one object per channel. */
  TdcAuxIO _auxIO[NAuxIO] -> auxio  [[shape_method(auxio_shape)]];	/* Axiliary configurations, one object per channel. */
  TdcVetoIO _veto -> veto;

  /* Standard constructor */
  @init()  [[auto, inline]];

}


//------------------ TdcDataV1_Item ------------------
/* Base class for all Acqiris TDC data objects. */
@type TdcDataV1_Item
  [[value_type]]
{
  /* Enum for types of data objects. Comm means data object has TdcDataV1Common
              type, AuxIO means TdcDataV1Marker class, all others are for TdcDataV1Channel. */
  @enum Source (uint8_t) {
    Comm,
    Chan1,
    Chan2,
    Chan3,
    Chan4,
    Chan5,
    Chan6,
    AuxIO,
  }

  uint32_t _value -> value {	/* Value as integer number whiis composed of several bit fields. Do not use value directly,
                instead cast this object to one of the actual types and use corresponding methods. */
    uint32_t _bf_val:28 -> bf_val_;
    Source _bf_source:3 -> source;	/* Source of this data object, use returned enum to distinguish between actual 
                types of data objecs and cast appropriately. */
    uint8_t _bf_ovf:1 -> bf_ofv_;
  }

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ TdcDataV1Common ------------------
/* Class for the "common" TDC data object. */
@type TdcDataV1Common(TdcDataV1_Item)
  [[value_type]]
{

  /* Returns number of hits. */
  uint32_t nhits()
  [[language("C++")]] @{ return @self.bf_val_(); @}

  /* Returns overflow status. */
  uint8_t overflow()
  [[language("C++")]] @{ return @self.bf_ofv_(); @}
}


//------------------ TdcDataV1Channel ------------------
/* Class for the "channel" TDC data object. */
@type TdcDataV1Channel(TdcDataV1_Item)
  [[value_type]]
{

  /* Returns number of ticks. */
  uint32_t ticks()
  [[language("C++")]] @{ return @self.bf_val_(); @}

  /* Returns overflow status. */
  uint8_t overflow()
  [[language("C++")]] @{ return @self.bf_ofv_(); @}

  /* Ticks converted to time, tick resolution is 50 picosecond. */
  double time()
  [[language("C++")]] @{ return @self.bf_val_() * 50e-12; @}
}


//------------------ TdcDataV1Marker ------------------
/* Class for the "marker" TDC data object. */
@type TdcDataV1Marker(TdcDataV1_Item)
  [[value_type]]
{
  /* Enum for the type of marker. */
  @enum Type (int32_t) {
    AuxIOSwitch = 0,
    EventCntSwitch = 1,
    MemFullSwitch = 2,
    AuxIOMarker = 16,
  }


  /* Returns type of the marker. */
  Type type()
  [[language("C++")]] @{ return Type(@self.bf_val_()); @}
}


//------------------ TdcDataV1 ------------------
/* Acqiris TDS data object is a container for TdcDataV1_Item object (or their
            sub-types). */
@type TdcDataV1
  [[type_id(Id_AcqTdcData, 1)]]
{
  TdcDataV1_Item _data[*] -> data;	/* Access TDC data items. */
}
} //- @package Acqiris
