@package EpixSampler  {


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_EpixSamplerConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  uint32_t _version -> version;
  uint32_t _runTrigDelay -> runTrigDelay;
  uint32_t _daqTrigDelay -> daqTrigDelay;
  uint32_t _daqSetting -> daqSetting;
  uint32_t _adcClkHalfT -> adcClkHalfT;
  uint32_t _adcPipelineDelay -> adcPipelineDelay;
  uint32_t _digitalCardId0 -> digitalCardId0;
  uint32_t _digitalCardId1 -> digitalCardId1;
  uint32_t _analogCardId0 -> analogCardId0;
  uint32_t _analogCardId1 -> analogCardId1;
  uint32_t _numberOfChannels -> numberOfChannels;
  uint32_t _samplesPerChannel -> samplesPerChannel;
  uint32_t _baseClockFrequency -> baseClockFrequency;
  uint32_t _bitControls {
    uint8_t _z1:8;
    uint8_t _testPatternEnable:1 -> testPatternEnable;
    uint32_t _z2:23;
  }

  double sampleInterval_sec()
  [[language("C++")]] @{
    double v=0;
    for (unsigned r=baseClockFrequency(); r!=0; r>>=4)
      v += 10*(r & 0xf);
    return double(adcClkHalfT())*2.e-3/v;
  @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ ElementV1 ------------------
@type ElementV1
  [[type_id(Id_EpixSamplerElement, 1)]]
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
    uint16_t _acqCount:16 -> acqCount;
  }
  uint32_t _frameNumber -> frameNumber;
  uint32_t _ticks -> ticks;
  uint32_t _fiducials -> fiducials;
  uint32_t _z0;
  uint32_t _z1;
  uint32_t _z2;
  uint16_t _frame[@config.numberOfChannels()][@config.samplesPerChannel()] -> frame;
  uint16_t _temperatures[@config.numberOfChannels()] -> temperatures;
  uint32_t _lastWord -> lastWord;
}
} //- @package EpixSampler
