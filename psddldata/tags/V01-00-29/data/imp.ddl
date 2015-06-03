@package Imp  {


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_ImpConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  @const int32_t MaxNumberOfSamples = 0x3ff;

  @enum Registers (int32_t) {
    Range,
    Cal_range,
    Reset,
    Bias_data,
    Cal_data,
    BiasDac_data,
    Cal_strobe,
    NumberOfSamples,
    TrigDelay,
    Adc_delay,
    NumberOfRegisters,
  }

  uint32_t _range -> range;
  uint32_t _calRange -> calRange;
  uint32_t _reset -> reset;
  uint32_t _biasData -> biasData;
  uint32_t _calData -> calData;
  uint32_t _biasDacData -> biasDacData;
  uint32_t _calStrobe -> calStrobe;
  uint32_t _numberOfSamples -> numberOfSamples;
  uint32_t _trigDelay -> trigDelay;
  uint32_t _adcDelay -> adcDelay;

  /* Constructor with value for each attribute */
  @init()  [[auto, inline]];

}


//------------------ Sample ------------------
@type Sample
  [[value_type]]
{
  @const int32_t channelsPerDevice = 4;

  uint16_t _channels[4] -> channels;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ LaneStatus ------------------
@type LaneStatus
  [[value_type]]
{
  uint32_t _value {
    uint8_t _usLinkErrCount:4 -> linkErrCount;
    uint8_t _usLinkDownCount:4 -> linkDownCount;
    uint8_t _usCellErrCount:4 -> cellErrCount;
    uint8_t _usRxCount:4 -> rxCount;
    uint8_t _usLocLinked:1 -> locLinked;
    uint8_t _usRemLinked:1 -> remLinked;
    uint16_t _zeros:10 -> zeros;
    uint8_t _powersOkay:4 -> powersOkay;
  }

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ ElementV1 ------------------
@type ElementV1
  [[type_id(Id_ImpData, 1)]]
  [[pack(2)]]
  [[config(ConfigV1)]]
{
  uint32_t _first {
    uint8_t _vc:2 -> vc;
    uint8_t _z:4;
    uint8_t _lane:2 -> lane;
    uint32_t _tid:24;
  }
  uint32_t _second {
    uint32_t _sz:32;
  }
  uint32_t _frameNumber -> frameNumber;
  uint32_t _ticks;
  uint32_t _fiducials;
  uint32_t _range -> range;
  LaneStatus _laneStatus -> laneStatus;
  uint32_t _z;
  Sample _samples[@config.numberOfSamples()] -> samples;
}
} //- @package Imp
