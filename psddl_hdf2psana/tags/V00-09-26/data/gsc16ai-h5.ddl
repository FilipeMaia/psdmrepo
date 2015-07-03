@include "psddldata/gsc16ai.ddl";
@package Gsc16ai  {


//------------------ ConfigV1 ------------------
@h5schema ConfigV1
  [[version(0)]]
{
  @dataset config {
    @attribute voltageRange;
    @attribute firstChan;
    @attribute lastChan;
    @attribute inputMode;
    @attribute triggerMode;
    @attribute dataFormat;
    @attribute fps;
    @attribute autocalibEnable;
    @attribute timeTagEnable;
  }
}


//------------------ DataV1 ------------------
@h5schema DataV1
  [[version(0)]]
{
  @dataset channelValue;
  @dataset timestamps [[method(timestamp)]];
}
} //- @package Gsc16ai
