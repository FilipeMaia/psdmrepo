@include "psddldata/xtc.ddl" [[headers("pdsdata/xtc/DetInfo.hh")]];
@package EvrData  {


//------------------ PulseConfig ------------------
@type PulseConfig
  [[value_type]]
{
  @const int32_t Trigger_Shift = 0;
  @const int32_t Set_Shift = 8;
  @const int32_t Clear_Shift = 16;
  @const int32_t Polarity_Shift = 0;
  @const int32_t Map_Set_Ena_Shift = 1;
  @const int32_t Map_Reset_Ena_Shift = 2;
  @const int32_t Map_Trigger_Ena_Shift = 3;

  uint32_t _pulse -> pulse;	/* internal pulse generation channel */
  uint32_t _input_control -> _input_control_value {	/* Pulse input control */
    int16_t _bf_trigger:8 -> bf_trigger  [[private]];
    int16_t _bf_set:8 -> bf_set  [[private]];
    int16_t _bf_clear:8 -> bf_clear  [[private]];
  }
  uint32_t _output_control -> _output_control_value {	/* Pulse output control */
    uint8_t _bf_polarity:1 -> polarity;
    uint8_t _bf_map_set_enable:1 -> map_set_enable;
    uint8_t _bf_map_reset_enable:1 -> map_reset_enable;
    uint8_t _bf_map_trigger_enable:1 -> map_trigger_enable;
  }
  uint32_t _prescale -> prescale;	/* pulse event prescale */
  uint32_t _delay -> delay;	/* delay in 119MHz clks */
  uint32_t _width -> width;	/* width in 119MHz clks */

  int16_t trigger()
  [[language("C++")]] @{ return @self.bf_trigger()-1; @}

  int16_t set()
  [[language("C++")]] @{ return @self.bf_set()-1; @}

  int16_t clear()
  [[language("C++")]] @{ return @self.bf_clear()-1; @}

  /* Constructor which takes values for every attribute */
  @init(pulse -> _pulse, int16_t trigger [[method(trigger)]], int16_t set [[method(set)]], 
        int16_t clear [[method(clear)]], polarity -> _bf_polarity, map_set_enable -> _bf_map_set_enable, 
        map_reset_enable -> _bf_map_reset_enable, map_trigger_enable -> _bf_map_trigger_enable, 
        prescale -> _prescale, delay -> _delay, width -> _width)
    _bf_trigger(trigger+1), _bf_set(set+1), _bf_clear(clear+1)  [[inline]];

}


//------------------ PulseConfigV3 ------------------
@type PulseConfigV3
  [[value_type]]
  [[pack(4)]]
{
  uint16_t _u16PulseId -> pulseId;
  uint16_t _u16Polarity -> polarity;	/* 0 -> positive polarity , 1 -> negative polarity */
  uint32_t _u32Prescale -> prescale;	/* Clock divider */
  uint32_t _u32Delay -> delay;	/* Delay in 119MHz clks */
  uint32_t _u32Width -> width;	/* Width in 119MHz clks */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ EventCodeV3 ------------------
@type EventCodeV3
  [[value_type]]
  [[pack(4)]]
{
  uint16_t _u16Code -> code;
  uint16_t _u16MaskEventAttr {
    uint8_t _bf_isReadout:1 -> isReadout;
    uint8_t _bf_isTerminator:1 -> isTerminator;
  }
  uint32_t _u32MaskTrigger -> maskTrigger;
  uint32_t _u32MaskSet -> maskSet;
  uint32_t _u32MaskClear -> maskClear;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ EventCodeV4 ------------------
@type EventCodeV4
  [[value_type]]
  [[pack(4)]]
{
  uint16_t _u16Code -> code;
  uint16_t _u16MaskEventAttr {
    uint8_t _bf_isReadout:1 -> isReadout;
    uint8_t _bf_isTerminator:1 -> isTerminator;
  }
  uint32_t _u32ReportDelay -> reportDelay;
  uint32_t _u32ReportWidth -> reportWidth;
  uint32_t _u32MaskTrigger -> maskTrigger;
  uint32_t _u32MaskSet -> maskSet;
  uint32_t _u32MaskClear -> maskClear;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ EventCodeV5 ------------------
@type EventCodeV5
  [[value_type]]
  [[pack(4)]]
{
  @const int32_t DescSize = 16;

  uint16_t _u16Code -> code;
  uint16_t _u16MaskEventAttr {
    uint8_t _bf_isReadout:1 -> isReadout;
    uint8_t _bf_isCommand:1 -> isCommand;
    uint8_t _bf_isLatch:1 -> isLatch;
  }
  uint32_t _u32ReportDelay -> reportDelay;
  uint32_t _u32ReportWidth -> reportWidth;
  uint32_t _u32MaskTrigger -> maskTrigger;
  uint32_t _u32MaskSet -> maskSet;
  uint32_t _u32MaskClear -> maskClear;
  char _desc[DescSize] -> desc;

  uint32_t releaseCode()  [[inline]]
  [[language("C++")]] @{ return @self._u32ReportWidth; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ EventCodeV6 ------------------
@type EventCodeV6
  [[value_type]]
  [[pack(4)]]
{
  @const int32_t DescSize = 16;
  @const int32_t MaxReadoutGroup = 7;

  uint16_t _u16Code -> code;
  uint16_t _u16MaskEventAttr {
    uint8_t _bf_isReadout:1 -> isReadout;
    uint8_t _bf_isCommand:1 -> isCommand;
    uint8_t _bf_isLatch:1 -> isLatch;
  }
  uint32_t _u32ReportDelay -> reportDelay;
  uint32_t _u32ReportWidth -> reportWidth;
  uint32_t _u32MaskTrigger -> maskTrigger;
  uint32_t _u32MaskSet -> maskSet;
  uint32_t _u32MaskClear -> maskClear;
  char _desc[DescSize] -> desc;
  uint16_t _u16ReadGroup -> readoutGroup;

  uint32_t releaseCode()  [[inline]]
  [[language("C++")]] @{ return @self._u32ReportWidth; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ SrcEventCode ------------------
/* Describes configuration of self-contained event generator. */
@type SrcEventCode
  [[value_type]]
  [[pack(4)]]
{
  @const int32_t DescSize = 16;
  @const int32_t MaxReadoutGroup = 7;

  uint16_t _u16Code -> code;	/* Assigned eventcode. */
  uint16_t _u16rsvd;
  uint32_t _u32Period -> period;	/* Repetition period in 119 MHz counts or 0 for external source. */
  uint32_t _u32MaskTriggerP -> maskTriggerP;	/* Bit mask of persistent pulse triggers. */
  uint32_t _u32MaskTriggerR -> maskTriggerR;	/* Bit mask of running pulse triggers. */
  char _desc[DescSize] -> desc;	/* Optional description. */
  uint16_t _u16ReadGroup -> readoutGroup;	/* Assigned readout group. */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ OutputMap ------------------
@type OutputMap
  [[value_type]]
{
  @enum Source (int16_t) {
    Pulse,
    DBus,
    Prescaler,
    Force_High,
    Force_Low,
  }
  @enum Conn (int16_t) {
    FrontPanel,
    UnivIO,
  }

  uint32_t _v -> value {
    Source _bf_source:8 -> source;
    uint8_t _bf_source_id:8 -> source_id;
    Conn _bf_conn:8 -> conn;
    uint8_t _bf_conn_id:8 -> conn_id;
  }

  /* Returns encoded source value. */
  uint32_t map()
  [[language("C++")]] @{
    enum { Pulse_Offset=0, DBus_Offset=32, Prescaler_Offset=40 };
    unsigned src_id = source_id();
    switch(source()) {
    case Pulse     : return src_id + Pulse_Offset;
    case DBus      : return src_id + DBus_Offset;
    case Prescaler : return src_id + Prescaler_Offset;
    case Force_High: return 62;
    case Force_Low : return 63;
    }
    return 0;
  @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ OutputMapV2 ------------------
@type OutputMapV2
  [[value_type]]
{
  @enum Source (int16_t) {
    Pulse,
    DBus,
    Prescaler,
    Force_High,
    Force_Low,
  }
  @enum Conn (int16_t) {
    FrontPanel,
    UnivIO,
  }

  uint32_t _v -> value {
    Source _bf_source:4 -> source;
    uint8_t _bf_source_id:8 -> source_id;
    Conn _bf_conn:4 -> conn;
    uint8_t _bf_conn_id:8 -> conn_id;
    uint8_t _bf_module:8 -> module;
  }

  /* Returns encoded source value. */
  uint32_t map()
  [[language("C++")]] @{
    enum { Pulse_Offset=0, DBus_Offset=32, Prescaler_Offset=40 };
    unsigned src_id = source_id();
    switch(source()) {
    case Pulse     : return src_id + Pulse_Offset;
    case DBus      : return src_id + DBus_Offset;
    case Prescaler : return src_id + Prescaler_Offset;
    case Force_High: return 62;
    case Force_Low : return 63;
    }
    return 0;
  @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_EvrConfig, 1)]]
  [[config_type]]
{
  uint32_t _npulses -> npulses;
  uint32_t _noutputs -> noutputs;
  PulseConfig _pulses[@self._npulses] -> pulses  [[shape_method(pulses_shape)]];
  OutputMap _output_maps[@self._noutputs] -> output_maps  [[shape_method(output_maps_shape)]];

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ ConfigV2 ------------------
@type ConfigV2
  [[type_id(Id_EvrConfig, 2)]]
  [[config_type]]
{
  @const int32_t beamOn = 100;
  @const int32_t baseRate = 40;
  @const int32_t singleShot = 150;

  @enum RateCode (int16_t) {
    r120Hz,
    r60Hz,
    r30Hz,
    r10Hz,
    r5Hz,
    r1Hz,
    r0_5Hz,
    Single,
    NumberOfRates,
  }
  @enum BeamCode (int16_t) {
    Off,
    On,
  }

  uint32_t _opcode -> opcode;
  uint32_t _npulses -> npulses;
  uint32_t _noutputs -> noutputs;
  PulseConfig _pulses[@self._npulses] -> pulses  [[shape_method(pulses_shape)]];
  OutputMap _output_maps[@self._noutputs] -> output_maps;

  BeamCode beam()
  [[language("C++")]] @{ return (@self._opcode > beamOn) ? On : Off; @}

  RateCode rate()
  [[language("C++")]] @{ 
    return (@self._opcode < beamOn) ? RateCode(@self._opcode-baseRate) :
	((@self._opcode < singleShot) ? RateCode(@self._opcode-beamOn-baseRate) : Single);
  @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ ConfigV3 ------------------
@type ConfigV3
  [[type_id(Id_EvrConfig, 3)]]
  [[config_type]]
  [[pack(4)]]
{
  uint32_t _neventcodes -> neventcodes;
  uint32_t _npulses -> npulses;
  uint32_t _noutputs -> noutputs;
  EventCodeV3 _eventcodes[@self._neventcodes] -> eventcodes;
  PulseConfigV3 _pulses[@self._npulses] -> pulses;
  OutputMap _output_maps[@self._noutputs] -> output_maps;

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ ConfigV4 ------------------
@type ConfigV4
  [[type_id(Id_EvrConfig, 4)]]
  [[config_type]]
  [[pack(4)]]
{
  uint32_t _neventcodes -> neventcodes;
  uint32_t _npulses -> npulses;
  uint32_t _noutputs -> noutputs;
  EventCodeV4 _eventcodes[@self._neventcodes] -> eventcodes;
  PulseConfigV3 _pulses[@self._npulses] -> pulses;
  OutputMap _output_maps[@self._noutputs] -> output_maps;

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ SequencerEntry ------------------
@type SequencerEntry
  [[value_type]]
{
  uint32_t _value {
    uint32_t _bf_delay:24 -> delay;
    uint32_t _bf_eventcode:8 -> eventcode;
  }

  /* Standard constructor */
  @init(eventcode -> _bf_eventcode, delay -> _bf_delay)  [[inline]];

}


//------------------ SequencerConfigV1 ------------------
@type SequencerConfigV1
{
  @enum Source (int32_t) {
    r120Hz,
    r60Hz,
    r30Hz,
    r10Hz,
    r5Hz,
    r1Hz,
    r0_5Hz,
    Disable,
  }

  uint32_t _source {
    Source _bf_sync_source:8 -> sync_source;
    Source _bf_beam_source:8 -> beam_source;
  }
  uint32_t _length -> length;
  uint32_t _cycles -> cycles;
  SequencerEntry _entries[@self.length()] -> entries;

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ ConfigV5 ------------------
@type ConfigV5
  [[type_id(Id_EvrConfig, 5)]]
  [[config_type]]
  [[pack(4)]]
{
  @const int32_t MaxPulses = 32;
  @const int32_t EvrOutputs = 10;

  uint32_t _neventcodes -> neventcodes;
  uint32_t _npulses -> npulses;
  uint32_t _noutputs -> noutputs;
  EventCodeV5 _eventcodes[@self._neventcodes] -> eventcodes;
  PulseConfigV3 _pulses[@self._npulses] -> pulses;
  OutputMap _output_maps[@self._noutputs] -> output_maps;
  SequencerConfigV1 _seq_config -> seq_config;

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ ConfigV6 ------------------
@type ConfigV6
  [[type_id(Id_EvrConfig, 6)]]
  [[config_type]]
  [[pack(4)]]
{
  /* Maximum pulses in the system */
  @const int32_t MaxPulses = 256;
  /* Maximum outputs in the system */
  @const int32_t MaxOutputs = 256;

  uint32_t _neventcodes -> neventcodes;
  uint32_t _npulses -> npulses;
  uint32_t _noutputs -> noutputs;
  EventCodeV5 _eventcodes[@self._neventcodes] -> eventcodes;
  PulseConfigV3 _pulses[@self._npulses] -> pulses;
  OutputMapV2 _output_maps[@self._noutputs] -> output_maps;
  SequencerConfigV1 _seq_config -> seq_config;

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ ConfigV7 ------------------
@type ConfigV7
  [[type_id(Id_EvrConfig, 7)]]
  [[config_type]]
  [[pack(4)]]
{
  /* Maximum pulses in the system */
  @const int32_t MaxPulses = 256;
  /* Maximum outputs in the system */
  @const int32_t MaxOutputs = 256;

  uint32_t _neventcodes -> neventcodes;
  uint32_t _npulses -> npulses;
  uint32_t _noutputs -> noutputs;
  EventCodeV6 _eventcodes[@self._neventcodes] -> eventcodes;
  PulseConfigV3 _pulses[@self._npulses] -> pulses;
  OutputMapV2 _output_maps[@self._noutputs] -> output_maps;
  SequencerConfigV1 _seq_config -> seq_config;

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ SrcConfigV1 ------------------
/* Describes configuration of self-contained event generator. */
@type SrcConfigV1
  [[type_id(Id_EvsConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  /* Maximum pulses in the system */
  @const int32_t MaxPulses = 12;
  /* Maximum outputs in the system */
  @const int32_t MaxOutputs = 12;

  uint32_t _neventcodes -> neventcodes;
  uint32_t _npulses -> npulses;
  uint32_t _noutputs -> noutputs;
  SrcEventCode _eventcodes[@self._neventcodes] -> eventcodes;
  PulseConfigV3 _pulses[@self._npulses] -> pulses;
  OutputMapV2 _output_maps[@self._noutputs] -> output_maps;

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ FIFOEvent ------------------
@type FIFOEvent
  [[value_type]]
{
  uint32_t _timestampHigh -> timestampHigh; /* 119 MHz timestamp (fiducial) */
  uint32_t _timestampLow -> timestampLow;   /* 360 Hz timestamp */
  uint32_t _eventCode -> eventCode;         /* event code (range 0-255) */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ DataV3 ------------------
@type DataV3
  [[type_id(Id_EvrData, 3)]]
{
  uint32_t _u32NumFifoEvents -> numFifoEvents;  /* length of FIFOEvent list */
  FIFOEvent _fifoEvents[@self._u32NumFifoEvents] -> fifoEvents;  /* FIFOEvent list */

  /* Standard constructor */
  @init()  [[auto, inline]];

  /* Constructor writing numFifoEvents only. */
  @init(u32NumFifoEvents -> _u32NumFifoEvents)  [[inline]];

}


//------------------ DataV4 ------------------
@type DataV4
  [[type_id(Id_EvrData, 4)]]
{
  uint32_t _u32NumFifoEvents -> numFifoEvents;  /* length of FIFOEvent list */
  FIFOEvent _fifoEvents[@self._u32NumFifoEvents] -> fifoEvents; /* FIFOEvent list */

  /* Returns 1 if the opcode is present in the EVR FIFO, otherwise 0. */
  uint8_t present(uint8_t opcode)
  [[language("C++")]] @{
    uint32_t size = @self.numFifoEvents();
    for (uint32_t ii = 0; ii < size; ii++) {
      if (@self.fifoEvents()[ii].eventCode() == opcode) {
        return 1;
      }
    }
    return 0;
  @}

  /* Standard constructor */
  @init()  [[auto, inline]];

  /* Constructor writing numFifoEvents only. */
  @init(u32NumFifoEvents -> _u32NumFifoEvents)  [[inline]];

}


//------------------ IOChannel ------------------
@type IOChannel
  [[value_type]]
{
  @const int32_t NameLength = 12;
  @const int32_t MaxInfos = 8;

  char _name[NameLength] -> name;
  uint32_t _ninfo -> ninfo;
  Pds.DetInfo _info[MaxInfos] -> infos;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ IOConfigV1 ------------------
@type IOConfigV1
  [[type_id(Id_EvrIOConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  uint16_t _conn;
  uint16_t _nchannels -> nchannels;
  IOChannel _channels[@self._nchannels] -> channels;

  EvrData.OutputMap.Conn conn()
  [[language("C++")]] @{ return OutputMap::Conn(@self._conn); @}

  /* Standard constructor */
  @init(conn -> _conn [[method(conn)]], nchannels -> _nchannels)  [[inline]];

}

//------------------ IOChannelV2 ------------------
@type IOChannelV2
  [[value_type]]
{
  @const int32_t NameLength = 64;
  @const int32_t MaxInfos = 16;

  /*  Output connector */
  OutputMapV2 _output           -> output;

  /*  Name of channel */
  char        _name[NameLength] -> name;

  /*  Number of Detectors connected */
  uint32_t _ninfo -> ninfo;

  /*  List of Detectors connected */
  Pds.DetInfo _info[MaxInfos] -> infos;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ IOConfigV2 ------------------
@type IOConfigV2
  [[type_id(Id_EvrIOConfig, 2)]]
  [[config_type]]
  [[pack(4)]]
{
  /* Number of Configured output channels */
  uint32_t    _nchannels -> nchannels;

  /* List of Configured output channels */
  IOChannelV2 _channels[@self._nchannels] -> channels;

  /* Standard constructor */
  @init(nchannels -> _nchannels)  [[inline]];

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];
}
} //- @package EvrData
