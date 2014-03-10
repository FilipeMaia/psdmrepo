@include "psddldata/rayonix.ddl";
@package Rayonix  {


//------------------ ConfigV1 ------------------
@h5schema ConfigV1
  [[version(0)]]
{
  @dataset config {
    @attribute binning_f;
    @attribute binning_s;
    @attribute exposure;
    @attribute trigger;
    @attribute rawMode;
    @attribute darkFlag;
    @attribute readoutMode;
    @attribute deviceID;
  }
}
} //- @package Rayonix
