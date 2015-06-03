@package Princeton  {


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_PrincetonConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  uint32_t _uWidth -> width;
  uint32_t _uHeight -> height;
  uint32_t _uOrgX -> orgX;
  uint32_t _uOrgY -> orgY;
  uint32_t _uBinX -> binX;
  uint32_t _uBinY -> binY;
  float _f32ExposureTime -> exposureTime;
  float _f32CoolingTemp -> coolingTemp;
  uint32_t _u32ReadoutSpeedIndex -> readoutSpeedIndex;
  uint16_t _u16ReadoutEventCode -> readoutEventCode;
  uint16_t _u16DelayMode -> delayMode;

  /* Total size in bytes of the Frame object including header and pixel data, 
            this returns the size of FrameV1 object, do not use this config type with FrameV2 */
  uint32_t frameSize()
  [[language("C++")]] @{ return 8 + @self.numPixels()*2; @}

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

  /* Standard constructor */
  @init()  [[auto, inline]];

}


//------------------ ConfigV2 ------------------
@type ConfigV2
  [[type_id(Id_PrincetonConfig, 2)]]
  [[config_type]]
  [[pack(4)]]
{
  uint32_t _uWidth -> width;
  uint32_t _uHeight -> height;
  uint32_t _uOrgX -> orgX;
  uint32_t _uOrgY -> orgY;
  uint32_t _uBinX -> binX;
  uint32_t _uBinY -> binY;
  float _f32ExposureTime -> exposureTime;
  float _f32CoolingTemp -> coolingTemp;
  uint16_t _u16GainIndex -> gainIndex;
  uint16_t _u16ReadoutSpeedIndex -> readoutSpeedIndex;
  uint16_t _u16ReadoutEventCode -> readoutEventCode;
  uint16_t _u16DelayMode -> delayMode;

  /* Total size in bytes of the Frame object including header and pixel data, 
            this returns the size of FrameV1 object, do not use this config type with FrameV2 */
  uint32_t frameSize()
  [[language("C++")]] @{ return 8 + @self.numPixels()*2; @}

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

  /* Standard constructor */
  @init()  [[auto, inline]];

}


//------------------ ConfigV3 ------------------
@type ConfigV3
  [[type_id(Id_PrincetonConfig, 3)]]
  [[config_type]]
  [[pack(4)]]
{
  uint32_t _uWidth -> width;
  uint32_t _uHeight -> height;
  uint32_t _uOrgX -> orgX;
  uint32_t _uOrgY -> orgY;
  uint32_t _uBinX -> binX;
  uint32_t _uBinY -> binY;
  float _f32ExposureTime -> exposureTime;
  float _f32CoolingTemp -> coolingTemp;
  uint8_t _u8GainIndex -> gainIndex;
  uint8_t _u8ReadoutSpeedIndex -> readoutSpeedIndex;
  uint16_t _u16ExposureEventCode -> exposureEventCode;
  uint32_t _u32NumDelayShots -> numDelayShots;

  /* Total size in bytes of the Frame object including header and pixel data, 
            this returns the size of FrameV1 object, do not use this config type with FrameV2 */
  uint32_t frameSize()
  [[language("C++")]] @{ return 8 + @self.numPixels()*2; @}

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

  /* Standard constructor */
  @init()  [[auto, inline]];

}


//------------------ ConfigV4 ------------------
@type ConfigV4
  [[type_id(Id_PrincetonConfig, 4)]]
  [[config_type]]
  [[pack(4)]]
{
  uint32_t _uWidth -> width;
  uint32_t _uHeight -> height;
  uint32_t _uOrgX -> orgX;
  uint32_t _uOrgY -> orgY;
  uint32_t _uBinX -> binX;
  uint32_t _uBinY -> binY;
  uint32_t _u32MaskedHeight -> maskedHeight;
  uint32_t _u32KineticHeight -> kineticHeight;
  float _f32VsSpeed -> vsSpeed;
  float _f32ExposureTime -> exposureTime;
  float _f32CoolingTemp -> coolingTemp;
  uint8_t _u8GainIndex -> gainIndex;
  uint8_t _u8ReadoutSpeedIndex -> readoutSpeedIndex;
  uint16_t _u16ExposureEventCode -> exposureEventCode;
  uint32_t _u32NumDelayShots -> numDelayShots;

  /* Total size in bytes of the Frame object including header and pixel data, 
            this returns the size of FrameV1 object, do not use this config type with FrameV2 */
  uint32_t frameSize()
  [[language("C++")]] @{ return 8 + @self.numPixels()*2; @}

  /* calculate frame X size in pixels based on the current ROI and binning settings */
  uint32_t numPixelsX()  [[inline]]
  [[language("C++")]] @{ return (@self.width() + @self.binX() - 1) / @self.binX(); @}

  /* calculate frame Y size in pixels based on the current ROI and binning settings */
  uint32_t numPixelsY()  [[inline]]
  [[language("C++")]] @{ return (@self.height()+ @self.binY() - 1) / @self.binY(); @}

  /* calculate total frame size in pixels based on the current ROI and binning settings */
  uint32_t numPixels()  [[inline]]
  [[language("C++")]] @{ return ((@self.width() + @self.binX()-1)/ @self.binX() )*((@self.height()+ @self.binY()-1)/ @self.binY() ); @}

  /* Standard constructor */
  @init()  [[auto, inline]];

}


//------------------ ConfigV5 ------------------
@type ConfigV5
  [[type_id(Id_PrincetonConfig, 5)]]
  [[config_type]]
  [[pack(4)]]
{
  uint32_t _uWidth -> width;
  uint32_t _uHeight -> height;
  uint32_t _uOrgX -> orgX;
  uint32_t _uOrgY -> orgY;
  uint32_t _uBinX -> binX;
  uint32_t _uBinY -> binY;
  float _f32ExposureTime -> exposureTime;
  float _f32CoolingTemp -> coolingTemp;
  uint16_t _u16GainIndex -> gainIndex;
  uint16_t _u16ReadoutSpeedIndex -> readoutSpeedIndex;
  uint32_t _u32MaskedHeight -> maskedHeight;
  uint32_t _u32KineticHeight -> kineticHeight;
  float _f32VsSpeed -> vsSpeed;
  int16_t _i16InfoReportInterval -> infoReportInterval;
  uint16_t _u16ExposureEventCode -> exposureEventCode;
  uint32_t _u32NumDelayShots -> numDelayShots;

  /* Total size in bytes of the Frame object including header and pixel data, 
            this returns the size of FrameV2 object, do not use this config type with FrameV1 */
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
  [[type_id(Id_PrincetonFrame, 1)]]
  [[pack(4)]]
  [[config(ConfigV1, ConfigV2, ConfigV3, ConfigV4, ConfigV5)]]
{
  uint32_t _iShotIdStart -> shotIdStart;
  float _fReadoutTime -> readoutTime;
  uint16_t _data[@config.numPixelsY()][@config.numPixelsX()] -> data;
}


//------------------ FrameV2 ------------------
@type FrameV2
  [[type_id(Id_PrincetonFrame, 2)]]
  [[pack(4)]]
  [[config(ConfigV1, ConfigV2, ConfigV3, ConfigV4, ConfigV5)]]
{
  uint32_t _iShotIdStart -> shotIdStart;
  float _fReadoutTime -> readoutTime;
  float _fTemperature -> temperature;
  uint16_t _data[@config.numPixelsY()][@config.numPixelsX()] -> data;

  /* Constructor which takes values for each attribute */
  @init(shotIdStart -> _iShotIdStart, fReadoutTime -> _fReadoutTime, fTemperature -> _fTemperature)  [[inline]];

}


//------------------ InfoV1 ------------------
@type InfoV1
  [[type_id(Id_PrincetonInfo, 1)]]
  [[value_type]]
{
  float _fTemperature -> temperature;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}
} //- @package Princeton
