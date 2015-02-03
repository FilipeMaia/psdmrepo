@package OceanOptics  {


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_OceanOpticsConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  float _f32ExposureTime -> exposureTime;
  double _lfWaveLenCalibCoeff[4] -> waveLenCalib;
  double _lfNonlinCorrectCoeff[8] -> nonlinCorrect;
  double _fStrayLightConstant -> strayLightConstant;

  /* Construct from exposure time */
  @init(f32ExposureTime -> _f32ExposureTime)  [[inline]];

  /* Construct from all attributes */
  @init()  [[auto, inline]];
}

//------------------ ConfigV2 ------------------
@type ConfigV2
  [[type_id(Id_OceanOpticsConfig, 2)]]
  [[config_type]]
  [[pack(4)]]
{
  float    _f32ExposureTime         -> exposureTime;
  int32_t  _iDeviceType             -> deviceType;
  double   _lfWaveLenCalibCoeff[4]  -> waveLenCalib;
  double   _lfNonlinCorrectCoeff[8] -> nonlinCorrect;
  double   _fStrayLightConstant     -> strayLightConstant;

  /* Construct from exposure time */
  @init(f32ExposureTime -> _f32ExposureTime)  [[inline]];

  /* Construct from all attributes */
  @init()  [[auto, inline]];

}



//------------------ timespec64 ------------------
@type timespec64
  [[value_type]]
  [[pack(4)]]
{
  uint64_t _tv_sec -> tv_sec;
  uint64_t _tv_nsec -> tv_nsec;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}

//------------------ DataV1 ------------------
@type DataV1
  [[type_id(Id_OceanOpticsData, 1)]]
  [[pack(4)]]
  [[config(ConfigV1, ConfigV2)]]
{
  @const int32_t iDataReadSize = 8192;
  @const int32_t iNumPixels = 3840;
  @const int32_t iActivePixelIndex = 22;

  uint16_t lu16Spetra[iNumPixels] -> data;
  uint64_t _u64FrameCounter -> frameCounter;
  uint64_t _u64NumDelayedFrames -> numDelayedFrames;
  uint64_t _u64NumDiscardFrames -> numDiscardFrames;
  timespec64 _tsTimeFrameStart -> timeFrameStart;
  timespec64 _tsTimeFrameFirstData -> timeFrameFirstData;
  timespec64 _tsTimeFrameEnd -> timeFrameEnd;
  int32_t _i32Version;
  int8_t _i8NumSpectraInData -> numSpectraInData;
  int8_t _i8NumSpectraInQueue -> numSpectraInQueue;
  int8_t _i8NumSpectraUnused -> numSpectraUnused;
  int8_t _iReserved1;

  double durationOfFrame()  [[inline]]
  [[language("C++")]] @{
    return @self.timeFrameEnd().tv_sec() - @self.timeFrameStart().tv_sec() +
  (@self.timeFrameEnd().tv_nsec() - @self.timeFrameStart().tv_nsec()) * 1e-9;
  @}

  double nonlinerCorrected(uint32_t iPixel)
  [[language("C++")]] @{
    double fRawValue = (double) (@self.data()[iPixel] ^ 0x2000);
    const ndarray<const double, 1>& corr = @config.nonlinCorrect();
    return fRawValue / (
  corr[0] + fRawValue *
       (corr[1] + fRawValue *
       (corr[2] + fRawValue *
       (corr[3] + fRawValue *
       (corr[4] + fRawValue *
       (corr[5] + fRawValue *
       (corr[6] + fRawValue *
        corr[7])))))));
  @}
}

//------------------ DataV2 ------------------
@type DataV2
  [[type_id(Id_OceanOpticsData, 2)]]
  [[pack(4)]]
  [[config(ConfigV2)]]
{
  @const int32_t iDataReadSize = 4608;
  @const int32_t iNumPixels = 2048;
  @const int32_t iActivePixelIndex = 0;

  uint16_t lu16Spetra[iNumPixels] -> data;
  uint64_t _u64FrameCounter -> frameCounter;
  uint64_t _u64NumDelayedFrames -> numDelayedFrames;
  uint64_t _u64NumDiscardFrames -> numDiscardFrames;
  timespec64 _tsTimeFrameStart -> timeFrameStart;
  timespec64 _tsTimeFrameFirstData -> timeFrameFirstData;
  timespec64 _tsTimeFrameEnd -> timeFrameEnd;
  int32_t _i32Version;
  int8_t _i8NumSpectraInData -> numSpectraInData;
  int8_t _i8NumSpectraInQueue -> numSpectraInQueue;
  int8_t _i8NumSpectraUnused -> numSpectraUnused;
  int8_t _iReserved1;

  double durationOfFrame()  [[inline]]
  [[language("C++")]] @{
    return @self.timeFrameEnd().tv_sec() - @self.timeFrameStart().tv_sec() +
  (@self.timeFrameEnd().tv_nsec() - @self.timeFrameStart().tv_nsec()) * 1e-9;
  @}

  double nonlinerCorrected(uint32_t iPixel)
  [[language("C++")]] @{
    double fRawValue = (double) (@self.data()[iPixel]);
    const ndarray<const double, 1>& corr = @config.nonlinCorrect();
    return fRawValue / (
  corr[0] + fRawValue *
       (corr[1] + fRawValue *
       (corr[2] + fRawValue *
       (corr[3] + fRawValue *
       (corr[4] + fRawValue *
       (corr[5] + fRawValue *
       (corr[6] + fRawValue *
        corr[7])))))));
  @}
}

//------------------ DataV3 ------------------
@type DataV3
  [[type_id(Id_OceanOpticsData, 3)]]
  [[pack(4)]]
  [[config(ConfigV2)]]
{
  @const int32_t iDataReadSize = 8192;
  @const int32_t iNumPixels = 3840;
  @const int32_t iActivePixelIndex = 22;

  uint16_t lu16Spetra[iNumPixels] -> data;
  uint64_t _u64FrameCounter -> frameCounter;
  uint64_t _u64NumDelayedFrames -> numDelayedFrames;
  uint64_t _u64NumDiscardFrames -> numDiscardFrames;
  timespec64 _tsTimeFrameStart -> timeFrameStart;
  timespec64 _tsTimeFrameFirstData -> timeFrameFirstData;
  timespec64 _tsTimeFrameEnd -> timeFrameEnd;
  int32_t _i32Version;
  int8_t _i8NumSpectraInData -> numSpectraInData;
  int8_t _i8NumSpectraInQueue -> numSpectraInQueue;
  int8_t _i8NumSpectraUnused -> numSpectraUnused;
  int8_t _iReserved1;

  double durationOfFrame()  [[inline]]
  [[language("C++")]] @{
    return @self.timeFrameEnd().tv_sec() - @self.timeFrameStart().tv_sec() +
  (@self.timeFrameEnd().tv_nsec() - @self.timeFrameStart().tv_nsec()) * 1e-9;
  @}

  double nonlinerCorrected(uint32_t iPixel)
  [[language("C++")]] @{
    double fRawValue = (double) @self.data()[iPixel];
    const ndarray<const double, 1>& corr = @config.nonlinCorrect();
    return fRawValue / (
  corr[0] + fRawValue *
       (corr[1] + fRawValue *
       (corr[2] + fRawValue *
       (corr[3] + fRawValue *
       (corr[4] + fRawValue *
       (corr[5] + fRawValue *
       (corr[6] + fRawValue *
        corr[7])))))));
  @}
}

} //- @package OceanOptics
