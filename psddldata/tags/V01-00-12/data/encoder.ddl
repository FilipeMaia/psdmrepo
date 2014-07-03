@package Encoder  {


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_EncoderConfig, 1)]]
  [[config_type]]
{
  @enum count_mode_type (uint32_t) {
    WRAP_FULL,
    LIMIT,
    HALT,
    WRAP_PRESET,
    COUNT_END,
  }
  @enum quad_mode (uint32_t) {
    CLOCK_DIR,
    X1,
    X2,
    X4,
    QUAD_END,
  }

  uint32_t _chan_num -> chan_num;
  count_mode_type _count_mode -> count_mode;
  quad_mode _quadrature_mode -> quadrature_mode;
  uint32_t _input_num -> input_num;
  uint32_t _input_rising -> input_rising;
  uint32_t _ticks_per_sec -> ticks_per_sec;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ ConfigV2 ------------------
@type ConfigV2
  [[type_id(Id_EncoderConfig, 2)]]
  [[config_type]]
{
  @enum count_mode_type (uint32_t) {
    WRAP_FULL,
    LIMIT,
    HALT,
    WRAP_PRESET,
    COUNT_END,
  }
  @enum quad_mode (uint32_t) {
    CLOCK_DIR,
    X1,
    X2,
    X4,
    QUAD_END,
  }

  uint32_t _chan_mask -> chan_mask;
  count_mode_type _count_mode -> count_mode;
  quad_mode _quadrature_mode -> quadrature_mode;
  uint32_t _input_num -> input_num;
  uint32_t _input_rising -> input_rising;
  uint32_t _ticks_per_sec -> ticks_per_sec;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ DataV1 ------------------
@type DataV1
  [[type_id(Id_EncoderData, 1)]]
{
  uint32_t _33mhz_timestamp -> timestamp;
  uint32_t _encoder_count -> encoder_count;

  /* Lower 24 bits of encoder_count as signed integer value. */
  int32_t value()
  [[language("C++")]] @{ return int(@self.encoder_count() << 8)/256; @}
}


//------------------ DataV2 ------------------
@type DataV2
  [[type_id(Id_EncoderData, 2)]]
{
  /* Number of encoders. */
  @const int32_t NEncoders = 3;

  uint32_t _33mhz_timestamp -> timestamp;
  uint32_t _encoder_count[NEncoders] -> encoder_count  [[shape_method(encoder_count_shape)]];

  /* Lower 24 bits of encoder_count as signed integer value. */
  int32_t value(uint32_t i)
  [[language("C++")]] @{ return int(@self.encoder_count()[i] << 8)/256; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}
} //- @package Encoder
