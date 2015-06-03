@package Lusi  {


//------------------ DiodeFexConfigV1 ------------------
@type DiodeFexConfigV1
  [[type_id(Id_DiodeFexConfig, 1)]]
  [[value_type]]
  [[config_type]]
  [[pack(4)]]
{
  @const int32_t NRANGES = 3;

  float _base[NRANGES] -> base;
  float _scale[NRANGES] -> scale;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ DiodeFexConfigV2 ------------------
@type DiodeFexConfigV2
  [[type_id(Id_DiodeFexConfig, 2)]]
  [[value_type]]
  [[config_type]]
  [[pack(4)]]
{
  @const int32_t NRANGES = 16;

  float _base[NRANGES] -> base;
  float _scale[NRANGES] -> scale;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ DiodeFexV1 ------------------
@type DiodeFexV1
  [[type_id(Id_DiodeFex, 1)]]
  [[value_type]]
  [[pack(4)]]
{
  float _value -> value;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ IpmFexConfigV1 ------------------
@type IpmFexConfigV1
  [[type_id(Id_IpmFexConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  @const int32_t NCHANNELS = 4;

  DiodeFexConfigV1 _diode[NCHANNELS] -> diode;
  float _xscale -> xscale;
  float _yscale -> yscale;

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ IpmFexConfigV2 ------------------
@type IpmFexConfigV2
  [[type_id(Id_IpmFexConfig, 2)]]
  [[config_type]]
  [[pack(4)]]
{
  @const int32_t NCHANNELS = 4;

  DiodeFexConfigV2 _diode[NCHANNELS] -> diode;
  float _xscale -> xscale;
  float _yscale -> yscale;

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ IpmFexV1 ------------------
@type IpmFexV1
  [[type_id(Id_IpmFex, 1)]]
  [[value_type]]
  [[pack(4)]]
{
  @const int32_t NCHANNELS = 4;

  float _channel[NCHANNELS] -> channel;
  float _sum -> sum;
  float _xpos -> xpos;
  float _ypos -> ypos;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ PimImageConfigV1 ------------------
@type PimImageConfigV1
  [[type_id(Id_PimImageConfig, 1)]]
  [[value_type]]
  [[config_type]]
  [[pack(4)]]
{
  float _xscale -> xscale;
  float _yscale -> yscale;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}
} //- @package Lusi
