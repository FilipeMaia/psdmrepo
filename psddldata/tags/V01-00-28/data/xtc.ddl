@package Pds [[external]] {


//------------------ ClockTime ------------------
@type ClockTime
  [[external]]
  [[value_type]]
  [[no_sizeof]]
{
  uint32_t _low -> nanoseconds;
  uint32_t _high -> seconds;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ DetInfo ------------------
@type DetInfo
  [[external]]
  [[value_type]]
  [[no_sizeof]]
{
  uint32_t _log -> log;
  uint32_t _phy -> phy;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ Src ------------------
@type Src
  [[external]]
  [[value_type]]
{
  uint32_t _log -> log;
  uint32_t _phy -> phy;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}

//------------------ TypeId ------------------
@type TypeId
  [[external]]
  [[value_type]]
{
  uint32_t _value -> value;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}
} //- @package Pds
