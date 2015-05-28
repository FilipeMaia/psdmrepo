@include "psddldata/encoder.ddl";
@package Encoder  {


//------------------ ConfigV1 ------------------
@h5schema ConfigV1
  [[version(0)]]
{
  @enum quad_mode {
    QUAD_END -> END,
  }
  @enum count_mode_type {
    COUNT_END -> END,
  }
  @dataset config {
    @attribute chan_num;
    @attribute count_mode;
    @attribute quadrature_mode;
    @attribute input_num;
    @attribute input_rising;
    @attribute ticks_per_sec;
  }
}


//------------------ ConfigV2 ------------------
@h5schema ConfigV2
  [[version(0)]]
{
  @enum quad_mode {
    QUAD_END -> END,
  }
  @enum count_mode_type {
    COUNT_END -> END,
  }
  @dataset config {
    @attribute chan_mask;
    @attribute count_mode;
    @attribute quadrature_mode;
    @attribute input_num;
    @attribute input_rising;
    @attribute ticks_per_sec;
  }
}


//------------------ DataV1 ------------------
@h5schema DataV1
  [[version(0)]]
{
  @dataset data {
    @attribute _33mhz_timestamp [[method(timestamp)]];
    @attribute encoder_count;
  }
}


//------------------ DataV2 ------------------
@h5schema DataV2
  [[version(0)]]
{
  @dataset data {
    @attribute _33mhz_timestamp [[method(timestamp)]];
    @attribute encoder_count;
  }
}
} //- @package Encoder
