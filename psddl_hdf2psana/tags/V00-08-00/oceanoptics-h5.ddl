@include "psddldata/oceanoptics.ddl";
@package OceanOptics  {


//------------------ ConfigV1 ------------------
@h5schema ConfigV1
  [[version(0)]]
{
  @dataset config {
    @attribute exposureTime;
    @attribute waveLenCalib;
    @attribute nonlinCorrect;
    @attribute strayLightConstant;
  }
}


//------------------ timespec64 ------------------
@h5schema timespec64
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute uint32_t seconds [[method(tv_sec)]];
    @attribute uint32_t nanoseconds [[method(tv_nsec)]];
  }
}


//------------------ DataV1 ------------------
@h5schema DataV1
  [[version(0)]]
{
  @dataset spectra [[method(data)]];
  @dataset data {
    @attribute frameCounter;
    @attribute numDelayedFrames;
    @attribute numDiscardFrames;
    @attribute timeFrameStart;
    @attribute timeFrameFirstData;
    @attribute timeFrameEnd;
    @attribute numSpectraInData;
    @attribute numSpectraInQueue;
    @attribute numSpectraUnused;
    @attribute durationOfFrame;
  }
}
} //- @package OceanOptics
