@include "psddldata/cspad2x2.ddl";
@package CsPad2x2  {


//------------------ CsPad2x2DigitalPotsCfg ------------------
@h5schema CsPad2x2DigitalPotsCfg
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute pots;
  }
}


//------------------ CsPad2x2ReadOnlyCfg ------------------
@h5schema CsPad2x2ReadOnlyCfg
  [[version(0)]]
  [[embedded]]
  [[default]]
{
}


//------------------ ProtectionSystemThreshold ------------------
@h5schema ProtectionSystemThreshold
  [[version(0)]]
  [[embedded]]
  [[default]]
{
}


//------------------ CsPad2x2GainMapCfg ------------------
@h5schema CsPad2x2GainMapCfg
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute gainMap;
  }
}


//------------------ ConfigV1QuadReg ------------------
@h5schema ConfigV1QuadReg
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute shiftSelect;
    @attribute edgeSelect;
    @attribute readClkSet;
    @attribute readClkHold;
    @attribute dataMode;
    @attribute prstSel;
    @attribute acqDelay;
    @attribute intTime;
    @attribute digDelay;
    @attribute ampIdle;
    @attribute injTotal;
    @attribute rowColShiftPer;
    @attribute ampReset;
    @attribute digCount;
    @attribute digPeriod;
    @attribute PeltierEnable;
    @attribute kpConstant;
    @attribute kiConstant;
    @attribute kdConstant;
    @attribute humidThold;
    @attribute setPoint;
    @attribute readOnly [[method(ro)]];
    @attribute digitalPots [[method(dp)]];
    @attribute gainMap [[method(gm)]];
  }
}


//------------------ ConfigV1 ------------------
@h5schema ConfigV1
  [[version(0)]]
{
  @dataset config {
    @attribute concentratorVersion;
    @attribute protectionThreshold;
    @attribute protectionEnable;
    @attribute inactiveRunMode;
    @attribute activeRunMode;
    @attribute testDataIndex [[method(tdi)]];
    @attribute payloadSize;
    @attribute badAsicMask;
    @attribute asicMask;
    @attribute roiMask;
    @attribute numAsicsRead;
    @attribute numAsicsStored;
    @attribute quad;
  }
}


//------------------ ConfigV2QuadReg ------------------
@h5schema ConfigV2QuadReg
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute shiftSelect;
    @attribute edgeSelect;
    @attribute readClkSet;
    @attribute readClkHold;
    @attribute dataMode;
    @attribute prstSel;
    @attribute acqDelay;
    @attribute intTime;
    @attribute digDelay;
    @attribute ampIdle;
    @attribute injTotal;
    @attribute rowColShiftPer;
    @attribute ampReset;
    @attribute digCount;
    @attribute digPeriod;
    @attribute PeltierEnable;
    @attribute kpConstant;
    @attribute kiConstant;
    @attribute kdConstant;
    @attribute humidThold;
    @attribute setPoint;
    @attribute biasTuning;
    @attribute pdpmndnmBalance;
    @attribute readOnly [[method(ro)]];
    @attribute digitalPots [[method(dp)]];
    @attribute gainMap [[method(gm)]];
  }
}


//------------------ ConfigV2 ------------------
@h5schema ConfigV2
  [[version(0)]]
{
  @dataset config {
    @attribute concentratorVersion;
    @attribute protectionThreshold;
    @attribute protectionEnable;
    @attribute inactiveRunMode;
    @attribute activeRunMode;
    @attribute runTriggerDelay;
    @attribute testDataIndex [[method(tdi)]];
    @attribute payloadSize;
    @attribute badAsicMask;
    @attribute asicMask;
    @attribute roiMask;
    @attribute numAsicsRead;
    @attribute numAsicsStored;
    @attribute quad;
  }
}


//------------------ ElementV1 ------------------
@h5schema ElementV1
  [[version(0)]]
  [[external("psddl_hdf2psana/cspad2x2.h")]]
{
}
} //- @package CsPad2x2
