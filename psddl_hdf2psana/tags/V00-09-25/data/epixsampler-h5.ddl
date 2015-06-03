@include "psddldata/epixsampler.ddl";
@package EpixSampler  {


//------------------ ConfigV1 ------------------
@h5schema ConfigV1
  [[version(0)]]
{
  @dataset config {
    @attribute version;
    @attribute runTrigDelay;
    @attribute daqTrigDelay;
    @attribute daqSetting;
    @attribute adcClkHalfT;
    @attribute adcPipelineDelay;
    @attribute digitalCardId0;
    @attribute digitalCardId1;
    @attribute analogCardId0;
    @attribute analogCardId1;
    @attribute numberOfChannels;
    @attribute samplesPerChannel;
    @attribute baseClockFrequency;
    @attribute testPatternEnable;
  }
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
  @dataset temperatures;
  @dataset frame;
}
} //- @package EpixSampler
