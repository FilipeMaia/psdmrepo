@include "psddldata/epix.ddl";
@package Epix  {


//------------------ AsicConfigV1 ------------------
@h5schema AsicConfigV1
  [[version(0)]]
  [[embedded]]
  [[default]]
{
} 

//------------------ Asic10kConfigV1 ------------------
@h5schema Asic10kConfigV1
  [[version(0)]]
  [[embedded]]
  [[default]]
{
}

//------------------ Asic100aConfigV1 ------------------
@h5schema Asic100aConfigV1
  [[version(0)]]
  [[embedded]]
  [[default]]
{
}

//------------------ ElementV1 ------------------
@h5schema ElementV1
  [[version(0)]]
{
  @dataset data {
    @attribute vc;
    @attribute lane;
    @attribute acqCount;
    @attribute frameNumber;
    @attribute ticks;
    @attribute fiducials;
    @attribute lastWord;
  }
  @dataset frame;
  @dataset excludedRows [[zero_dims]];
  @dataset temperatures;
}

//------------------ ElementV2 ------------------
@h5schema ElementV2
  [[version(0)]]
{
  @dataset data {
    @attribute vc;
    @attribute lane;
    @attribute acqCount;
    @attribute frameNumber;
    @attribute ticks;
    @attribute fiducials;
    @attribute lastWord;
  }
  @dataset frame;
  @dataset calibrationRows [[zero_dims]];
  @dataset environmentalRows [[zero_dims]];
  @dataset temperatures;
}
} //- @package Epix
