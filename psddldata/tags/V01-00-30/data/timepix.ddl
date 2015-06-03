@package Timepix  {


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_TimepixConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  @const int32_t ChipCount = 4;

  @enum ReadoutSpeed (uint8_t) {
    ReadoutSpeed_Slow = 0,
    ReadoutSpeed_Fast = 1,
  }
  @enum TriggerMode (uint8_t) {
    TriggerMode_ExtPos = 0,
    TriggerMode_ExtNeg = 1,
    TriggerMode_Soft = 2,
  }

  ReadoutSpeed _readoutSpeed -> readoutSpeed;
  TriggerMode _triggerMode -> triggerMode;
  int16_t _pad;
  int32_t _shutterTimeout -> shutterTimeout;
  int32_t _dac0Ikrum -> dac0Ikrum;
  int32_t _dac0Disc -> dac0Disc;
  int32_t _dac0Preamp -> dac0Preamp;
  int32_t _dac0BufAnalogA -> dac0BufAnalogA;
  int32_t _dac0BufAnalogB -> dac0BufAnalogB;
  int32_t _dac0Hist -> dac0Hist;
  int32_t _dac0ThlFine -> dac0ThlFine;
  int32_t _dac0ThlCourse -> dac0ThlCourse;
  int32_t _dac0Vcas -> dac0Vcas;
  int32_t _dac0Fbk -> dac0Fbk;
  int32_t _dac0Gnd -> dac0Gnd;
  int32_t _dac0Ths -> dac0Ths;
  int32_t _dac0BiasLvds -> dac0BiasLvds;
  int32_t _dac0RefLvds -> dac0RefLvds;
  int32_t _dac1Ikrum -> dac1Ikrum;
  int32_t _dac1Disc -> dac1Disc;
  int32_t _dac1Preamp -> dac1Preamp;
  int32_t _dac1BufAnalogA -> dac1BufAnalogA;
  int32_t _dac1BufAnalogB -> dac1BufAnalogB;
  int32_t _dac1Hist -> dac1Hist;
  int32_t _dac1ThlFine -> dac1ThlFine;
  int32_t _dac1ThlCourse -> dac1ThlCourse;
  int32_t _dac1Vcas -> dac1Vcas;
  int32_t _dac1Fbk -> dac1Fbk;
  int32_t _dac1Gnd -> dac1Gnd;
  int32_t _dac1Ths -> dac1Ths;
  int32_t _dac1BiasLvds -> dac1BiasLvds;
  int32_t _dac1RefLvds -> dac1RefLvds;
  int32_t _dac2Ikrum -> dac2Ikrum;
  int32_t _dac2Disc -> dac2Disc;
  int32_t _dac2Preamp -> dac2Preamp;
  int32_t _dac2BufAnalogA -> dac2BufAnalogA;
  int32_t _dac2BufAnalogB -> dac2BufAnalogB;
  int32_t _dac2Hist -> dac2Hist;
  int32_t _dac2ThlFine -> dac2ThlFine;
  int32_t _dac2ThlCourse -> dac2ThlCourse;
  int32_t _dac2Vcas -> dac2Vcas;
  int32_t _dac2Fbk -> dac2Fbk;
  int32_t _dac2Gnd -> dac2Gnd;
  int32_t _dac2Ths -> dac2Ths;
  int32_t _dac2BiasLvds -> dac2BiasLvds;
  int32_t _dac2RefLvds -> dac2RefLvds;
  int32_t _dac3Ikrum -> dac3Ikrum;
  int32_t _dac3Disc -> dac3Disc;
  int32_t _dac3Preamp -> dac3Preamp;
  int32_t _dac3BufAnalogA -> dac3BufAnalogA;
  int32_t _dac3BufAnalogB -> dac3BufAnalogB;
  int32_t _dac3Hist -> dac3Hist;
  int32_t _dac3ThlFine -> dac3ThlFine;
  int32_t _dac3ThlCourse -> dac3ThlCourse;
  int32_t _dac3Vcas -> dac3Vcas;
  int32_t _dac3Fbk -> dac3Fbk;
  int32_t _dac3Gnd -> dac3Gnd;
  int32_t _dac3Ths -> dac3Ths;
  int32_t _dac3BiasLvds -> dac3BiasLvds;
  int32_t _dac3RefLvds -> dac3RefLvds;

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ ConfigV2 ------------------
@type ConfigV2
  [[type_id(Id_TimepixConfig, 2)]]
  [[config_type]]
  [[pack(4)]]
{
  @const int32_t ChipCount = 4;
  @const int32_t ChipNameMax = 16;
  @const int32_t PixelThreshMax = 4*256*256;

  @enum ReadoutSpeed (uint8_t) {
    ReadoutSpeed_Slow = 0,
    ReadoutSpeed_Fast = 1,
  }
  @enum TriggerMode (uint8_t) {
    TriggerMode_ExtPos = 0,
    TriggerMode_ExtNeg = 1,
    TriggerMode_Soft = 2,
  }

  ReadoutSpeed _readoutSpeed -> readoutSpeed;
  TriggerMode _triggerMode -> triggerMode;
  int16_t _pad;
  int32_t _timepixSpeed -> timepixSpeed;
  int32_t _dac0Ikrum -> dac0Ikrum;
  int32_t _dac0Disc -> dac0Disc;
  int32_t _dac0Preamp -> dac0Preamp;
  int32_t _dac0BufAnalogA -> dac0BufAnalogA;
  int32_t _dac0BufAnalogB -> dac0BufAnalogB;
  int32_t _dac0Hist -> dac0Hist;
  int32_t _dac0ThlFine -> dac0ThlFine;
  int32_t _dac0ThlCourse -> dac0ThlCourse;
  int32_t _dac0Vcas -> dac0Vcas;
  int32_t _dac0Fbk -> dac0Fbk;
  int32_t _dac0Gnd -> dac0Gnd;
  int32_t _dac0Ths -> dac0Ths;
  int32_t _dac0BiasLvds -> dac0BiasLvds;
  int32_t _dac0RefLvds -> dac0RefLvds;
  int32_t _dac1Ikrum -> dac1Ikrum;
  int32_t _dac1Disc -> dac1Disc;
  int32_t _dac1Preamp -> dac1Preamp;
  int32_t _dac1BufAnalogA -> dac1BufAnalogA;
  int32_t _dac1BufAnalogB -> dac1BufAnalogB;
  int32_t _dac1Hist -> dac1Hist;
  int32_t _dac1ThlFine -> dac1ThlFine;
  int32_t _dac1ThlCourse -> dac1ThlCourse;
  int32_t _dac1Vcas -> dac1Vcas;
  int32_t _dac1Fbk -> dac1Fbk;
  int32_t _dac1Gnd -> dac1Gnd;
  int32_t _dac1Ths -> dac1Ths;
  int32_t _dac1BiasLvds -> dac1BiasLvds;
  int32_t _dac1RefLvds -> dac1RefLvds;
  int32_t _dac2Ikrum -> dac2Ikrum;
  int32_t _dac2Disc -> dac2Disc;
  int32_t _dac2Preamp -> dac2Preamp;
  int32_t _dac2BufAnalogA -> dac2BufAnalogA;
  int32_t _dac2BufAnalogB -> dac2BufAnalogB;
  int32_t _dac2Hist -> dac2Hist;
  int32_t _dac2ThlFine -> dac2ThlFine;
  int32_t _dac2ThlCourse -> dac2ThlCourse;
  int32_t _dac2Vcas -> dac2Vcas;
  int32_t _dac2Fbk -> dac2Fbk;
  int32_t _dac2Gnd -> dac2Gnd;
  int32_t _dac2Ths -> dac2Ths;
  int32_t _dac2BiasLvds -> dac2BiasLvds;
  int32_t _dac2RefLvds -> dac2RefLvds;
  int32_t _dac3Ikrum -> dac3Ikrum;
  int32_t _dac3Disc -> dac3Disc;
  int32_t _dac3Preamp -> dac3Preamp;
  int32_t _dac3BufAnalogA -> dac3BufAnalogA;
  int32_t _dac3BufAnalogB -> dac3BufAnalogB;
  int32_t _dac3Hist -> dac3Hist;
  int32_t _dac3ThlFine -> dac3ThlFine;
  int32_t _dac3ThlCourse -> dac3ThlCourse;
  int32_t _dac3Vcas -> dac3Vcas;
  int32_t _dac3Fbk -> dac3Fbk;
  int32_t _dac3Gnd -> dac3Gnd;
  int32_t _dac3Ths -> dac3Ths;
  int32_t _dac3BiasLvds -> dac3BiasLvds;
  int32_t _dac3RefLvds -> dac3RefLvds;
  int32_t _driverVersion -> driverVersion;
  uint32_t _firmwareVersion -> firmwareVersion;
  uint32_t _pixelThreshSize -> pixelThreshSize;
  uint8_t _pixelThresh[PixelThreshMax] -> pixelThresh;
  char _chip0Name[ChipNameMax] -> chip0Name  [[shape_method(None)]];
  char _chip1Name[ChipNameMax] -> chip1Name  [[shape_method(None)]];
  char _chip2Name[ChipNameMax] -> chip2Name  [[shape_method(None)]];
  char _chip3Name[ChipNameMax] -> chip3Name  [[shape_method(None)]];
  int32_t _chip0ID -> chip0ID;
  int32_t _chip1ID -> chip1ID;
  int32_t _chip2ID -> chip2ID;
  int32_t _chip3ID -> chip3ID;

  int32_t chipCount()  [[inline]]
  [[language("C++")]] @{ return ChipCount; @}

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ ConfigV3 ------------------
@type ConfigV3
  [[type_id(Id_TimepixConfig, 3)]]
  [[config_type]]
  [[pack(4)]]
{
  @const int32_t ChipCount = 4;
  @const int32_t ChipNameMax = 16;
  @const int32_t PixelThreshMax = 4*256*256;

  @enum ReadoutSpeed (uint8_t) {
    ReadoutSpeed_Slow = 0,
    ReadoutSpeed_Fast = 1,
  }
  @enum TimepixMode (uint8_t) {
    TimepixMode_Counting = 0,
    TimepixMode_TOT = 1,
  }

  ReadoutSpeed _readoutSpeed -> readoutSpeed;
  TimepixMode _timepixMode -> timepixMode;
  int16_t _pad;
  int32_t _timepixSpeed -> timepixSpeed;
  int32_t _dac0Ikrum -> dac0Ikrum;
  int32_t _dac0Disc -> dac0Disc;
  int32_t _dac0Preamp -> dac0Preamp;
  int32_t _dac0BufAnalogA -> dac0BufAnalogA;
  int32_t _dac0BufAnalogB -> dac0BufAnalogB;
  int32_t _dac0Hist -> dac0Hist;
  int32_t _dac0ThlFine -> dac0ThlFine;
  int32_t _dac0ThlCourse -> dac0ThlCourse;
  int32_t _dac0Vcas -> dac0Vcas;
  int32_t _dac0Fbk -> dac0Fbk;
  int32_t _dac0Gnd -> dac0Gnd;
  int32_t _dac0Ths -> dac0Ths;
  int32_t _dac0BiasLvds -> dac0BiasLvds;
  int32_t _dac0RefLvds -> dac0RefLvds;
  int32_t _dac1Ikrum -> dac1Ikrum;
  int32_t _dac1Disc -> dac1Disc;
  int32_t _dac1Preamp -> dac1Preamp;
  int32_t _dac1BufAnalogA -> dac1BufAnalogA;
  int32_t _dac1BufAnalogB -> dac1BufAnalogB;
  int32_t _dac1Hist -> dac1Hist;
  int32_t _dac1ThlFine -> dac1ThlFine;
  int32_t _dac1ThlCourse -> dac1ThlCourse;
  int32_t _dac1Vcas -> dac1Vcas;
  int32_t _dac1Fbk -> dac1Fbk;
  int32_t _dac1Gnd -> dac1Gnd;
  int32_t _dac1Ths -> dac1Ths;
  int32_t _dac1BiasLvds -> dac1BiasLvds;
  int32_t _dac1RefLvds -> dac1RefLvds;
  int32_t _dac2Ikrum -> dac2Ikrum;
  int32_t _dac2Disc -> dac2Disc;
  int32_t _dac2Preamp -> dac2Preamp;
  int32_t _dac2BufAnalogA -> dac2BufAnalogA;
  int32_t _dac2BufAnalogB -> dac2BufAnalogB;
  int32_t _dac2Hist -> dac2Hist;
  int32_t _dac2ThlFine -> dac2ThlFine;
  int32_t _dac2ThlCourse -> dac2ThlCourse;
  int32_t _dac2Vcas -> dac2Vcas;
  int32_t _dac2Fbk -> dac2Fbk;
  int32_t _dac2Gnd -> dac2Gnd;
  int32_t _dac2Ths -> dac2Ths;
  int32_t _dac2BiasLvds -> dac2BiasLvds;
  int32_t _dac2RefLvds -> dac2RefLvds;
  int32_t _dac3Ikrum -> dac3Ikrum;
  int32_t _dac3Disc -> dac3Disc;
  int32_t _dac3Preamp -> dac3Preamp;
  int32_t _dac3BufAnalogA -> dac3BufAnalogA;
  int32_t _dac3BufAnalogB -> dac3BufAnalogB;
  int32_t _dac3Hist -> dac3Hist;
  int32_t _dac3ThlFine -> dac3ThlFine;
  int32_t _dac3ThlCourse -> dac3ThlCourse;
  int32_t _dac3Vcas -> dac3Vcas;
  int32_t _dac3Fbk -> dac3Fbk;
  int32_t _dac3Gnd -> dac3Gnd;
  int32_t _dac3Ths -> dac3Ths;
  int32_t _dac3BiasLvds -> dac3BiasLvds;
  int32_t _dac3RefLvds -> dac3RefLvds;
  int8_t _dacBias -> dacBias;
  int8_t _flags -> flags;
  int16_t _pad2;
  int32_t _driverVersion -> driverVersion;
  uint32_t _firmwareVersion -> firmwareVersion;
  uint32_t _pixelThreshSize -> pixelThreshSize;
  uint8_t _pixelThresh[PixelThreshMax] -> pixelThresh;
  char _chip0Name[ChipNameMax] -> chip0Name  [[shape_method(None)]];
  char _chip1Name[ChipNameMax] -> chip1Name  [[shape_method(None)]];
  char _chip2Name[ChipNameMax] -> chip2Name  [[shape_method(None)]];
  char _chip3Name[ChipNameMax] -> chip3Name  [[shape_method(None)]];
  int32_t _chip0ID -> chip0ID;
  int32_t _chip1ID -> chip1ID;
  int32_t _chip2ID -> chip2ID;
  int32_t _chip3ID -> chip3ID;

  int32_t chipCount()  [[inline]]
  [[language("C++")]] @{ return ChipCount; @}

  /* Constructor with a value for each argument */
  @init()  [[auto, inline]];

}


//------------------ DataV1 ------------------
@type DataV1
  [[type_id(Id_TimepixData, 1)]]
{
  @const int32_t Height = 512;
  @const int32_t Width = 512;
  @const int32_t Depth = 14;
  @const int32_t DepthBytes = 2;
  @const int32_t MaxPixelValue = 11810;

  uint32_t _timestamp -> timestamp;
  uint16_t _frameCounter -> frameCounter;
  uint16_t _lostRows -> lostRows;
  uint16_t _data[Height][Width] -> data;

  uint32_t width()  [[inline]]
  [[language("C++")]] @{ return Width; @}

  uint32_t height()  [[inline]]
  [[language("C++")]] @{ return Height; @}

  uint32_t depth()  [[inline]]
  [[language("C++")]] @{ return Depth; @}

  uint32_t depth_bytes()  [[inline]]
  [[language("C++")]] @{ return DepthBytes; @}

  /* Size of the image data in bytes. */
  uint32_t data_size()  [[inline]]
  [[language("C++")]] @{ return @self.width() * @self.height() * @self.depth_bytes(); @}

  /* Constructor which takes values for scalar attributes */
  @init(timestamp -> _timestamp, frameCounter -> _frameCounter, lostRows -> _lostRows)  [[inline]];

}


//------------------ DataV2 ------------------
@type DataV2
  [[type_id(Id_TimepixData, 2)]]
{
  @const int32_t Depth = 14;
  @const int32_t MaxPixelValue = 11810;

  uint16_t _width -> width;	/* Pixels per row */
  uint16_t _height -> height;	/* Pixels per column */
  uint32_t _timestamp -> timestamp;	/* hardware timestamp */
  uint16_t _frameCounter -> frameCounter;	/* hardware frame counter */
  uint16_t _lostRows -> lostRows;	/* lost row count */
  uint16_t _data[@self.height()][@self.width()] -> data;

  uint32_t depth()  [[inline]]
  [[language("C++")]] @{ return Depth; @}

  uint32_t depth_bytes()  [[inline]]
  [[language("C++")]] @{ return (Depth+7)/8; @}

  /* Size of the image data in bytes. */
  uint32_t data_size()  [[inline]]
  [[language("C++")]] @{ return @self.width() * @self.height() * @self.depth_bytes(); @}

  /* Constructor with a value for each argument */
  @init(width -> _width, height -> _height, timestamp -> _timestamp, frameCounter -> _frameCounter, lostRows -> _lostRows)  [[inline]];

  /* Special conversion constructor from DataV1, implementation is defined in a separate file */
  @init(DataV1 datav1)  [[external]];

}
} //- @package Timepix
