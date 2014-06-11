@include "psddldata/ipimb.ddl";
@package Ipimb  {


//------------------ ConfigV1 ------------------
@h5schema ConfigV1
  [[version(0)]]
{
  @dataset config {
    @attribute triggerCounter;
    @attribute serialID;
    @attribute chargeAmpRange;
    @attribute capacitorValue[4] [[method(capacitorValues)]];
    @attribute calibrationRange;
    @attribute resetLength;
    @attribute resetDelay;
    @attribute chargeAmpRefVoltage;
    @attribute calibrationVoltage;
    @attribute diodeBias;
    @attribute status;
    @attribute errors;
    @attribute calStrobeLength;
    @attribute trigDelay;
  }
}


//------------------ ConfigV2 ------------------
@h5schema ConfigV2
  [[version(0)]]
{
  @dataset config {
    @attribute triggerCounter;
    @attribute serialID;
    @attribute chargeAmpRange;
    @attribute capacitorValue[4] [[method(capacitorValues)]];
    @attribute calibrationRange;
    @attribute resetLength;
    @attribute resetDelay;
    @attribute chargeAmpRefVoltage;
    @attribute calibrationVoltage;
    @attribute diodeBias;
    @attribute status;
    @attribute errors;
    @attribute calStrobeLength;
    @attribute trigDelay;
    @attribute trigPsDelay;
    @attribute adcDelay;
  }
}
} //- @package Ipimb
