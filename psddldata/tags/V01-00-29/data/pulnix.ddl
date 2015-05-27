@package Pulnix  {


//------------------ TM6740ConfigV1 ------------------
@type TM6740ConfigV1
  [[type_id(Id_TM6740Config, 1)]]
  [[config_type]]
{
  @const int32_t Row_Pixels = 480;
  @const int32_t Column_Pixels = 640;

  @enum Depth (uint8_t) {
    Eight_bit,
    Ten_bit,
  }
  @enum Binning (uint8_t) {
    x1,
    x2,
    x4,
  }
  @enum LookupTable (uint8_t) {
    Gamma,
    Linear,
  }

  uint32_t _gain_a_b {
    uint16_t _bf_gain_a:16 -> gain_a;
    uint16_t _bf_gain_b:16 -> gain_b;
  }
  uint32_t _vref_shutter {
    uint16_t _bf_vref:16 -> vref;
    uint16_t _bf_shutter:16 -> shutter_width;
  }
  uint32_t _control {
    uint8_t _bf_gain_balance:1 -> gain_balance;
    Depth _bf_output_resolution:1 -> output_resolution;
    Binning _bf_horizontal_binning:2 -> horizontal_binning;
    Binning _bf_vertical_binning:2 -> vertical_binning;
    LookupTable _bf_lookuptable_mode:1 -> lookuptable_mode;
  }

  /* bit-depth of pixel counts */
  uint8_t output_resolution_bits()
  [[language("C++")]] @{ return @self.output_resolution() == Eight_bit ? 8 : 10; @}

  /* Constructor which takes values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ TM6740ConfigV2 ------------------
@type TM6740ConfigV2
  [[type_id(Id_TM6740Config, 2)]]
  [[config_type]]
{
  @const int32_t Row_Pixels = 480;
  @const int32_t Column_Pixels = 640;

  @enum Depth (int32_t) {
    Eight_bit,
    Ten_bit,
  }
  @enum Binning (int32_t) {
    x1,
    x2,
    x4,
  }
  @enum LookupTable (int32_t) {
    Gamma,
    Linear,
  }

  uint32_t _gain_a_b {
    uint16_t _bf_gain_a:16 -> gain_a;
    uint16_t _bf_gain_b:16 -> gain_b;
  }
  uint32_t _vref_shutter {
    uint16_t _bf_vref_a:10 -> vref_a;
    uint32_t _bf_pad0:6;
    uint16_t _bf_vref_b:10 -> vref_b;
  }
  uint32_t _control {
    uint8_t _bf_gain_balance:1 -> gain_balance;
    Depth _bf_output_resolution:1 -> output_resolution;
    Binning _bf_horizontal_binning:2 -> horizontal_binning;
    Binning _bf_vertical_binning:2 -> vertical_binning;
    LookupTable _bf_lookuptable_mode:1 -> lookuptable_mode;
  }

  /* bit-depth of pixel counts */
  uint8_t output_resolution_bits()
  [[language("C++")]] @{ return @self.output_resolution() == Eight_bit ? 8 : 10; @}

  /* Constructor which takes values for each attribute */
  @init()  [[auto, inline]];

}
} //- @package Pulnix
