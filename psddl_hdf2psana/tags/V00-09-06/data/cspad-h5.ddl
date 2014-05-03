@include "psddldata/cspad.ddl";
@package CsPad  {


//------------------ CsPadDigitalPotsCfg ------------------
@h5schema CsPadDigitalPotsCfg
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute pots;
  }
}


//------------------ CsPadReadOnlyCfg ------------------
@h5schema CsPadReadOnlyCfg
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


//------------------ CsPadGainMapCfg ------------------
@h5schema CsPadGainMapCfg
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
    @attribute readOnly [[method(ro)]];
    @attribute digitalPots [[method(dp)]];
    @attribute gainMap [[method(gm)]];
    @attribute shiftSelect;
    @attribute edgeSelect;
  }
}


//------------------ ConfigV2QuadReg ------------------
@h5schema ConfigV2QuadReg
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
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
    @attribute readOnly [[method(ro)]];
    @attribute digitalPots [[method(dp)]];
    @attribute gainMap [[method(gm)]];
    @attribute shiftSelect;
    @attribute edgeSelect;
  }
}


//------------------ ConfigV3QuadReg ------------------
@h5schema ConfigV3QuadReg
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
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
    @attribute biasTuning;
    @attribute pdpmndnmBalance;
    @attribute readOnly [[method(ro)]];
    @attribute digitalPots [[method(dp)]];
    @attribute gainMap [[method(gm)]];
    @attribute shiftSelect;
    @attribute edgeSelect;
  }
}


//------------------ ConfigV1 ------------------
@h5schema ConfigV1
  [[version(0)]]
{
  @dataset config {
    @attribute concentratorVersion;
    @attribute runDelay;
    @attribute eventCode;
    @attribute inactiveRunMode;
    @attribute activeRunMode;
    @attribute testDataIndex [[method(tdi)]];
    @attribute payloadPerQuad [[method(payloadSize)]];
    @attribute badAsicMask0;
    @attribute badAsicMask1;
    @attribute asicMask;
    @attribute quadMask;
    @attribute quads;
  }
}


//------------------ ConfigV2 ------------------
@h5schema ConfigV2
  [[version(0)]]
{
  @dataset config {
    @attribute concentratorVersion;
    @attribute runDelay;
    @attribute eventCode;
    @attribute inactiveRunMode;
    @attribute activeRunMode;
    @attribute testDataIndex [[method(tdi)]];
    @attribute payloadPerQuad [[method(payloadSize)]];
    @attribute badAsicMask0;
    @attribute badAsicMask1;
    @attribute asicMask;
    @attribute quadMask;
    @attribute roiMask [[method(roiMasks)]];
    @attribute quads;
    @attribute int8_t sections[MaxQuadsPerSensor][SectorsPerQuad] [[external("psddl_hdf2psana/cspad.h")]];
  }
}


//------------------ ConfigV3 ------------------
@h5schema ConfigV3
  [[version(0)]]
{
  @dataset config {
    @attribute concentratorVersion;
    @attribute runDelay;
    @attribute eventCode;
    @attribute protectionEnable;
    @attribute inactiveRunMode;
    @attribute activeRunMode;
    @attribute testDataIndex [[method(tdi)]];
    @attribute payloadPerQuad [[method(payloadSize)]];
    @attribute badAsicMask0;
    @attribute badAsicMask1;
    @attribute asicMask;
    @attribute quadMask;
    @attribute roiMask [[method(roiMasks)]];
    @attribute protectionThresholds;
    @attribute quads;
    @attribute int8_t sections[MaxQuadsPerSensor][SectorsPerQuad] [[external("psddl_hdf2psana/cspad.h")]];
  }
}


//------------------ ConfigV4 ------------------
@h5schema ConfigV4
  [[version(0)]]
{
  @dataset config {
    @attribute concentratorVersion;
    @attribute runDelay;
    @attribute eventCode;
    @attribute protectionEnable;
    @attribute inactiveRunMode;
    @attribute activeRunMode;
    @attribute testDataIndex [[method(tdi)]];
    @attribute payloadPerQuad [[method(payloadSize)]];
    @attribute badAsicMask0;
    @attribute badAsicMask1;
    @attribute asicMask;
    @attribute quadMask;
    @attribute roiMask [[method(roiMasks)]];
    @attribute protectionThresholds;
    @attribute quads;
    @attribute int8_t sections[MaxQuadsPerSensor][SectorsPerQuad] [[external("psddl_hdf2psana/cspad.h")]];
  }
}


//------------------ ConfigV5 ------------------
@h5schema ConfigV5
  [[version(0)]]
{
  @dataset config {
    @attribute concentratorVersion;
    @attribute runDelay;
    @attribute eventCode;
    @attribute protectionEnable;
    @attribute inactiveRunMode;
    @attribute activeRunMode;
    @attribute internalTriggerDelay;
    @attribute testDataIndex [[method(tdi)]];
    @attribute payloadPerQuad [[method(payloadSize)]];
    @attribute badAsicMask0;
    @attribute badAsicMask1;
    @attribute asicMask;
    @attribute quadMask;
    @attribute roiMask [[method(roiMasks)]];
    @attribute protectionThresholds;
    @attribute quads;
    @attribute int8_t sections[MaxQuadsPerSensor][SectorsPerQuad] [[external("psddl_hdf2psana/cspad.h")]];
  }
}


//------------------ ElementV1 ------------------
@h5schema ElementV1
  [[version(0)]]
  [[embedded]]
{
  @dataset element {
    @attribute virtual_channel;
    @attribute lane;
    @attribute tid;
    @attribute acq_count;
    @attribute op_code;
    @attribute quad;
    @attribute seq_count;
    @attribute ticks;
    @attribute fiducials;
    @attribute frame_type;
    @attribute sectionMask;
    @attribute sb_temp;
  }
  @dataset data;
}


//------------------ DataV1 ------------------
@h5schema DataV1
  [[version(0)]]
  [[external("psddl_hdf2psana/cspad.h")]]
{
}


//------------------ ElementV2 ------------------
@h5schema ElementV2
  [[version(0)]]
  [[embedded]]
{
  @dataset element {
    @attribute virtual_channel;
    @attribute lane;
    @attribute tid;
    @attribute acq_count;
    @attribute op_code;
    @attribute quad;
    @attribute seq_count;
    @attribute ticks;
    @attribute fiducials;
    @attribute frame_type;
    @attribute sectionMask;
    @attribute sb_temp;
  }
  @dataset data;
}


//------------------ DataV2 ------------------
@h5schema DataV2
  [[version(0)]]
  [[external("psddl_hdf2psana/cspad.h")]]
{
}
} //- @package CsPad
