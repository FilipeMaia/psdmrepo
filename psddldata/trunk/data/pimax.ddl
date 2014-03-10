@package Pimax  {

//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_PimaxConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  uint32_t _uWidth                  -> width;
  uint32_t _uHeight                 -> height;
  uint32_t _uOrgX                   -> orgX;
  uint32_t _uOrgY                   -> orgY;
  uint32_t _uBinX                   -> binX;
  uint32_t _uBinY                   -> binY;
  float    _f32ExposureTime         -> exposureTime;
  float    _f32CoolingTemp          -> coolingTemp;
  float    _f32ReadoutSpeed         -> readoutSpeed;
  uint16_t _u16GainIndex            -> gainIndex;
  uint16_t _u16IntensifierGain      -> intensifierGain;
  double   _f64GateDelay            -> gateDelay;
  double   _f64GateWidth            -> gateWidth;
  uint32_t _u32MaskedHeight         -> maskedHeight;
  uint32_t _u32KineticHeight        -> kineticHeight;
  float    _f32VsSpeed              -> vsSpeed;
  int16_t  _i16InfoReportInterval   -> infoReportInterval;
  uint16_t _u16ExposureEventCode    -> exposureEventCode;
  uint32_t _u32NumIntegrationShots  -> numIntegrationShots;

  /* Total size in bytes of the Frame object */
  uint32_t frameSize()
  [[language("C++")]] @{ return 12 + @self.numPixels()*2; @}

  /* calculate frame X size in pixels based on the current ROI and binning settings */
  uint32_t numPixelsX()  [[inline]]
  [[language("C++")]] @{ return (@self.width() + @self.binX() - 1) / @self.binX(); @}

  /* calculate frame Y size in pixels based on the current ROI and binning settings */
  uint32_t numPixelsY()  [[inline]]
  [[language("C++")]] @{ return (@self.height()+ @self.binY() - 1) / @self.binY(); @}

  /* calculate total frame size in pixels based on the current ROI and binning settings */
  uint32_t numPixels()  [[inline]]
  [[language("C++")]] @{
    return ((@self.width() + @self.binX()-1)/ @self.binX() )*((@self.height()+ @self.binY()-1)/ @self.binY() );
  @}

  /* Constructor which takes values for each attribute */
  @init()  [[auto, inline]];

  /* Constructor which takes values for each attribute */
  @init(width -> _uWidth, height -> _uHeight)  [[inline]];

}

//------------------ FrameV1 ------------------
@type FrameV1
  [[type_id(Id_PimaxFrame, 1)]]
  [[pack(4)]]
  [[config(ConfigV1)]]
{
  uint32_t  _iShotIdStart -> shotIdStart;
  float     _fReadoutTime -> readoutTime;
  float     _fTemperature -> temperature;
  uint16_t _data[@config.numPixelsY()][@config.numPixelsX()] -> data;

  /* Constructor with values for scalar attributes */
  @init(iShotIdStart -> _iShotIdStart, fReadoutTime -> _fReadoutTime, fTemperature -> _fTemperature)  [[inline]];

}

} //- @package Pimax
