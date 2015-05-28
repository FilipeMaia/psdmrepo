@include "psddldata/xtc.ddl" [[headers("pdsdata/xtc/ClockTime.hh")]];
@package Pds [[external]] {


//------------------ ClockTime ------------------
@h5schema ClockTime
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute nanoseconds;
    @attribute seconds;
  }
}


//------------------ DetInfo ------------------
@h5schema DetInfo
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute log;
    @attribute phy;
  }
}
} //- @package Pds
