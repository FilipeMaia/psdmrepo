@package Rayonix  {


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_RayonixConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  @const int32_t DeviceIDMax = 40;

  @enum ReadoutMode (uint32_t) {
    Standard = 0,
    HighGain = 1,
    LowNoise = 2,
    EDR = 3,
  }

  uint8_t _binning_f -> binning_f;
  uint8_t _binning_s -> binning_s;
  int16_t _pad;
  uint32_t _exposure -> exposure;
  uint32_t _trigger -> trigger;
  uint16_t _rawMode -> rawMode;
  uint16_t _darkFlag -> darkFlag;
  ReadoutMode _readoutMode -> readoutMode;
  char _deviceID[DeviceIDMax] -> deviceID  [[shape_method(None)]];

  /* Constructor with a value for each argument */
  @init()  [[auto, inline]];

}


//------------------ ConfigV2 ------------------
@type ConfigV2
  [[type_id(Id_RayonixConfig, 2)]]
  [[config_type]]
  [[pack(4)]]
{
  @const int32_t DeviceIDMax = 40;

  @enum ReadoutMode (uint32_t) {
    Unknown = 0,
    Standard = 1,
    HighGain = 2,
    LowNoise = 3,
    HDR = 4,
  }

  uint8_t _binning_f -> binning_f;
  uint8_t _binning_s -> binning_s;
  int16_t _testPattern -> testPattern;
  uint32_t _exposure -> exposure;
  uint32_t _trigger -> trigger;
  uint16_t _rawMode -> rawMode;
  uint16_t _darkFlag -> darkFlag;
  ReadoutMode _readoutMode -> readoutMode;
  char _deviceID[DeviceIDMax] -> deviceID  [[shape_method(None)]];

  /* Constructor with a value for each argument */
  @init()  [[auto, inline]];

}
} //- @package Rayonix
