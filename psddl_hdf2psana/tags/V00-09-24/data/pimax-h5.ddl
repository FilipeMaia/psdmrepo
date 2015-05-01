@include "psddldata/pimax.ddl";
@package Pimax {


//------------------ FrameV1 ------------------
@h5schema FrameV1
  [[version(0)]]
{
  @dataset frame {
    @attribute shotIdStart;
    @attribute readoutTime;
    @attribute temperature;
  }
  @dataset data;
}
} //- @package Pimax
