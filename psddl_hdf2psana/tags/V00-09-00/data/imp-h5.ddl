@include "psddldata/imp.ddl";
@package Imp  {


//------------------ Sample ------------------
@h5schema Sample
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute channels;
  }
}


//------------------ LaneStatus ------------------
@h5schema LaneStatus
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute linkErrCount;
    @attribute linkDownCount;
    @attribute cellErrCount;
    @attribute rxCount;
    @attribute locLinked;
    @attribute remLinked;
    @attribute zeros;
    @attribute powersOkay;
  }
}
} //- @package Imp
