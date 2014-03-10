@package UsdUsb  {


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_UsdUsbConfig, 1)]]
  [[config_type]]
{
  @const int32_t NCHANNELS = 4;

  @enum Count_Mode (uint32_t) {
    WRAP_FULL,
    LIMIT,
    HALT,
    WRAP_PRESET,
  }
  @enum Quad_Mode (uint32_t) {
    CLOCK_DIR,
    X1,
    X2,
    X4,
  }

  Count_Mode _count_mode[NCHANNELS] -> counting_mode;
  Quad_Mode _quad_mode[NCHANNELS] -> quadrature_mode;

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ DataV1 ------------------
@type DataV1
  [[type_id(Id_UsdUsbData, 1)]]
{
  @const int32_t Encoder_Inputs = 4;
  @const int32_t Analog_Inputs = 4;
  @const int32_t Digital_Inputs = 8;

  uint8_t _header[6];
  uint8_t _din -> digital_in;
  uint8_t _estop;
  uint32_t _timestamp -> timestamp;
  uint32_t _count[Encoder_Inputs];
  uint8_t _status[4] -> status;
  uint16_t _ain[Analog_Inputs] -> analog_in;

  /* Return lower 24 bits of _count array as signed integer values. */
  int32_t[] encoder_count()
  [[language("C++")]] @{
    unsigned shape[1] = {Encoder_Inputs};
    ndarray<int32_t,1> res(shape);
    for (unsigned i=0; i!=Encoder_Inputs; ++i) res[i]=int(@self._count[i] << 8)/256;
    return res;
  @}
}
} //- @package UsdUsb
