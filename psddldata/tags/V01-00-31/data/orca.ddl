@package Orca  {


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_OrcaConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  @const int32_t Row_Pixels = 2048;
  @const int32_t Column_Pixels = 2048;

  @enum ReadoutMode (uint8_t) {
    x1,
    x2,
    x4,
    Subarray,
  }
  @enum Cooling (uint8_t) {
    Off,
    On,
    Max,
  }

  uint32_t _options {
    ReadoutMode _bf_readoutMode:2 -> mode;
    Cooling _bf_cooling:2 -> cooling;
    uint8_t _bf_defect_pixel_correction_enabled:1 -> defect_pixel_correction_enabled;
  }
  uint32_t _rows -> rows;

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}
} //- @package Orca
