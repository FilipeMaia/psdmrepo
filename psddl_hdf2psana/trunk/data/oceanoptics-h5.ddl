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


//------------------ ConfigV2 ------------------
@h5schema ConfigV2
  [[version(0)]]
{
  @dataset config {
    @attribute exposureTime;
    @attribute deviceType;
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
  [[external("psddl_hdf2psana/oceanoptics.h")]]
{
}


//------------------ DataV2 ------------------
@h5schema DataV2
  [[version(0)]]
  [[external("psddl_hdf2psana/oceanoptics.h")]]
{
}

//------------------ DataV3 ------------------
@h5schema DataV3
  [[version(0)]]
  [[external("psddl_hdf2psana/oceanoptics.h")]]
{
}

} //- @package OceanOptics
