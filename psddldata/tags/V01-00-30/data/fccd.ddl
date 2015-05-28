@package FCCD  {


//------------------ FccdConfigV1 ------------------
@type FccdConfigV1
  [[type_id(Id_FccdConfig, 1)]]
  [[config_type]]
{
  @const int32_t Row_Pixels = 500;
  @const int32_t Column_Pixels = 576;
  @const int32_t Trimmed_Row_Pixels = 480;
  @const int32_t Trimmed_Column_Pixels = 480;

  @enum Depth (int32_t) {
    Sixteen_bit = 16,
  }
  @enum Output_Source (int32_t) {
    Output_FIFO = 0,
    Output_Pattern4 = 4,
  }

  uint16_t _u16OutputMode -> outputMode;

  uint32_t width()
  [[language("C++")]] @{ return Column_Pixels; @}

  uint32_t height()
  [[language("C++")]] @{ return Row_Pixels; @}

  uint32_t trimmedWidth()
  [[language("C++")]] @{ return Trimmed_Column_Pixels; @}

  uint32_t trimmedHeight()
  [[language("C++")]] @{ return Trimmed_Row_Pixels; @}
}


//------------------ FccdConfigV2 ------------------
@type FccdConfigV2
  [[type_id(Id_FccdConfig, 2)]]
  [[config_type]]
  [[pack(4)]]
{
  @const int32_t Row_Pixels = 500;
  @const int32_t Column_Pixels = 576 * 2;
  @const int32_t Trimmed_Row_Pixels = 480;
  @const int32_t Trimmed_Column_Pixels = 480;
  @const int32_t NVoltages = 17;
  @const int32_t NWaveforms = 15;

  @enum Depth (int32_t) {
    Eight_bit = 8,
    Sixteen_bit = 16,
  }
  @enum Output_Source (int32_t) {
    Output_FIFO = 0,
    Test_Pattern1 = 1,
    Test_Pattern2 = 2,
    Test_Pattern3 = 3,
    Test_Pattern4 = 4,
  }

  uint16_t _outputMode -> outputMode;
  uint8_t _ccdEnable -> ccdEnable;
  uint8_t _focusMode -> focusMode;
  uint32_t _exposureTime -> exposureTime;
  float _dacVoltage[NVoltages] -> dacVoltages;
  uint16_t _waveform[NWaveforms] -> waveforms;

  uint32_t width()
  [[language("C++")]] @{ return Column_Pixels; @}

  uint32_t height()
  [[language("C++")]] @{ return Row_Pixels; @}

  uint32_t trimmedWidth()
  [[language("C++")]] @{ return Trimmed_Column_Pixels; @}

  uint32_t trimmedHeight()
  [[language("C++")]] @{ return Trimmed_Row_Pixels; @}

  /* Constructor which takes values for each attribute */
  @init()  [[auto, inline]];

}
} //- @package FCCD
