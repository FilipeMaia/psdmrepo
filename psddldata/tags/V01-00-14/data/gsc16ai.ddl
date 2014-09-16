@package Gsc16ai  {


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_Gsc16aiConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  @const int32_t LowestChannel = 0;
  @const int32_t HighestChannel = 15;
  @const int32_t LowestFps = 1;
  @const int32_t HighestFps = 120;

  @enum InputMode (uint16_t) {
    InputMode_Differential = 0,
    InputMode_Zero = 1,
    InputMode_Vref = 2,
  }
  @enum VoltageRange (uint16_t) {
    VoltageRange_10V = 0,
    VoltageRange_5V,
    VoltageRange_2_5V,
  }
  @enum TriggerMode (uint16_t) {
    TriggerMode_ExtPos = 0,
    TriggerMode_ExtNeg,
    TriggerMode_IntClk,
  }
  @enum DataFormat (uint16_t) {
    DataFormat_TwosComplement = 0,
    DataFormat_OffsetBinary,
  }

  VoltageRange _voltageRange -> voltageRange;
  uint16_t _firstChan -> firstChan;
  uint16_t _lastChan -> lastChan;
  InputMode _inputMode -> inputMode;
  TriggerMode _triggerMode -> triggerMode;
  DataFormat _dataFormat -> dataFormat;
  uint16_t _fps -> fps;
  uint8_t _autocalibEnable -> autocalibEnable;
  uint8_t _timeTagEnable -> timeTagEnable;

  uint16_t numChannels()  [[inline]]
  [[language("C++")]] @{ return @self.lastChan() - @self.firstChan() + 1; @}

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ DataV1 ------------------
@type DataV1
  [[type_id(Id_Gsc16aiData, 1)]]
  [[config(ConfigV1)]]
{
  uint16_t _timestamp[3] -> timestamp;
  uint16_t _channelValue[@config.numChannels()] -> channelValue;
}
} //- @package Gsc16ai
