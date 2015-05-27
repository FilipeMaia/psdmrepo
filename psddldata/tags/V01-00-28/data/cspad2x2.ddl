@package CsPad2x2  {
  /* Defines number of quadrants in a CsPad2x2 device. */
  @const int32_t QuadsPerSensor = 1;
  /* Total number of ASICs in one quadrant. */
  @const int32_t ASICsPerQuad = 4;
  /* Number of rows per readout bank? */
  @const int32_t RowsPerBank = 26;
  /* Number of full readout banks per one ASIC? */
  @const int32_t FullBanksPerASIC = 7;
  /* Number of readout banks per one ASIC? */
  @const int32_t BanksPerASIC = 8;
  /* Number of columns readout by single ASIC. */
  @const int32_t ColumnsPerASIC = 185;
  /* Maximum number of rows readout by single ASIC. */
  @const int32_t MaxRowsPerASIC = 194;
  /* Number of POTs? per single quadrant. */
  @const int32_t PotsPerQuad = 80;
  /* Total number of 2x2s in single quadrant. */
  @const int32_t TwoByTwosPerQuad = 1;
  /* Total number of sectors (2x1) per single quadrant. */
  @const int32_t SectorsPerQuad = 2;
  /* Enum specifying different running modes. */
  @enum RunModes (int32_t) {
    NoRunning,
    RunButDrop,
    RunAndSendToRCE,
    RunAndSendTriggeredByTTL,
    ExternalTriggerSendToRCE,
    ExternalTriggerDrop,
    NumberOfRunModes,
  }
  /* Enum specifying different data collection modes. */
  @enum DataModes (int32_t) {
    normal = 0,
    shiftTest = 1,
    testData = 2,
    reserved = 3,
  }


//------------------ CsPad2x2DigitalPotsCfg ------------------
/* Class defining configuration for CsPad POTs? */
@type CsPad2x2DigitalPotsCfg
{
  uint8_t _pots[PotsPerQuad] -> pots;

  /* Standard constructor */
  @init()  [[auto]];

}


//------------------ CsPad2x2ReadOnlyCfg ------------------
/* Class defining read-only configuration. */
@type CsPad2x2ReadOnlyCfg
  [[value_type]]
{
  uint32_t _shiftTest -> shiftTest;
  uint32_t _version -> version;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ ProtectionSystemThreshold ------------------
@type ProtectionSystemThreshold
  [[value_type]]
{
  uint32_t _adcThreshold -> adcThreshold;
  uint32_t _pixelCountThreshold -> pixelCountThreshold;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ CsPad2x2GainMapCfg ------------------
/* Class defining ASIC gain map. */
@type CsPad2x2GainMapCfg
{
  uint16_t _gainMap[ColumnsPerASIC][MaxRowsPerASIC] -> gainMap;	/* Array with the gain map for single ASIC. */

  /* Standard constructor */
  @init()  [[auto]];

}


//------------------ ConfigV1QuadReg ------------------
/* Configuration data for single "quadrant" which for 2x2 means a single 2x2. */
@type ConfigV1QuadReg
{
  uint32_t _shiftSelect -> shiftSelect;
  uint32_t _edgeSelect -> edgeSelect;
  uint32_t _readClkSet -> readClkSet;
  uint32_t _readClkHold -> readClkHold;
  uint32_t _dataMode -> dataMode;
  uint32_t _prstSel -> prstSel;
  uint32_t _acqDelay -> acqDelay;
  uint32_t _intTime -> intTime;
  uint32_t _digDelay -> digDelay;
  uint32_t _ampIdle -> ampIdle;
  uint32_t _injTotal -> injTotal;
  uint32_t _rowColShiftPer -> rowColShiftPer;
  uint32_t _ampReset -> ampReset;
  uint32_t _digCount -> digCount;
  uint32_t _digPeriod -> digPeriod;
  uint32_t _PeltierEnable -> PeltierEnable;
  uint32_t _kpConstant -> kpConstant;
  uint32_t _kiConstant -> kiConstant;
  uint32_t _kdConstant -> kdConstant;
  uint32_t _humidThold -> humidThold;
  uint32_t _setPoint -> setPoint;
  CsPad2x2ReadOnlyCfg _readOnly -> ro;	/* read-only configuration */
  CsPad2x2DigitalPotsCfg _digitalPots -> dp;
  CsPad2x2GainMapCfg _gainMap -> gm;	/* Gain map. */

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ ConfigV1 ------------------
/* Configuration data for 2x2 CsPad device. */
@type ConfigV1
  [[type_id(Id_Cspad2x2Config, 1)]]
  [[config_type]]
{
  uint32_t _concentratorVersion -> concentratorVersion;
  ProtectionSystemThreshold _protectionThreshold -> protectionThreshold;
  uint32_t _protectionEnable -> protectionEnable;
  uint32_t _inactiveRunMode -> inactiveRunMode;
  uint32_t _activeRunMode -> activeRunMode;
  uint32_t _testDataIndex -> tdi;
  uint32_t _payloadPerQuad -> payloadSize;
  uint32_t _badAsicMask -> badAsicMask;
  uint32_t _AsicMask -> asicMask;
  uint32_t _roiMask -> roiMask;
  ConfigV1QuadReg _quad -> quad;

  uint32_t numAsicsRead()
  [[language("C++")]] @{ return 4; @}

  /* Number of ASICs in given quadrant */
  uint32_t numAsicsStored()
  [[language("C++")]] @{ return __builtin_popcount(@self.roiMask())*2; @}

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ ConfigV2QuadReg ------------------
/* Configuration data for single "quadrant" which for 2x2 means a single 2x2. */
@type ConfigV2QuadReg
{
  uint32_t _shiftSelect -> shiftSelect;
  uint32_t _edgeSelect -> edgeSelect;
  uint32_t _readClkSet -> readClkSet;
  uint32_t _readClkHold -> readClkHold;
  uint32_t _dataMode -> dataMode;
  uint32_t _prstSel -> prstSel;
  uint32_t _acqDelay -> acqDelay;
  uint32_t _intTime -> intTime;
  uint32_t _digDelay -> digDelay;
  uint32_t _ampIdle -> ampIdle;
  uint32_t _injTotal -> injTotal;
  uint32_t _rowColShiftPer -> rowColShiftPer;
  uint32_t _ampReset -> ampReset;
  uint32_t _digCount -> digCount;
  uint32_t _digPeriod -> digPeriod;
  uint32_t _PeltierEnable -> PeltierEnable;
  uint32_t _kpConstant -> kpConstant;
  uint32_t _kiConstant -> kiConstant;
  uint32_t _kdConstant -> kdConstant;
  uint32_t _humidThold -> humidThold;
  uint32_t _setPoint -> setPoint;
  uint32_t _biasTuning -> biasTuning;	/* bias tuning is used, but not written;
            2 bits per nibble, C2,C1,I5,I2;
            bit order rc00rc00rc00rc */
  uint32_t _pdpmndnmBalance -> pdpmndnmBalance;	/* pMOS and nMOS Displacement and Main;
            used but not written and not in GUI yet;
            hard-wired to zero in GUI;
            2 bits per nibble, bit order pd00pm00nd00nm */
  CsPad2x2ReadOnlyCfg _readOnly -> ro;	/* read-only configuration */
  CsPad2x2DigitalPotsCfg _digitalPots -> dp;
  CsPad2x2GainMapCfg _gainMap -> gm;	/* Gain map. */

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ ConfigV2 ------------------
/* Configuration data for 2x2 CsPad device. */
@type ConfigV2
  [[type_id(Id_Cspad2x2Config, 2)]]
  [[config_type]]
{
  uint32_t _concentratorVersion -> concentratorVersion;
  ProtectionSystemThreshold _protectionThreshold -> protectionThreshold;
  uint32_t _protectionEnable -> protectionEnable;
  uint32_t _inactiveRunMode -> inactiveRunMode;
  uint32_t _activeRunMode -> activeRunMode;
  uint32_t _runTriggerDelay -> runTriggerDelay;
  uint32_t _testDataIndex -> tdi;
  uint32_t _payloadPerQuad -> payloadSize;
  uint32_t _badAsicMask -> badAsicMask;
  uint32_t _AsicMask -> asicMask;
  uint32_t _roiMask -> roiMask;
  ConfigV2QuadReg _quad -> quad;

  uint32_t numAsicsRead()
  [[language("C++")]] @{ return 4; @}

  /* Number of ASICs in given quadrant */
  uint32_t numAsicsStored()
  [[language("C++")]] @{ return __builtin_popcount(@self.roiMask())*2; @}

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ ElementV1 ------------------
/* CsPad data from single 2x2 element. */
@type ElementV1
  [[type_id(Id_Cspad2x2Element, 1)]]
{
  /* Number of the elements in _sbtemp array. */
  @const int32_t Nsbtemp = 4;

  uint32_t _word0 {
    uint32_t _bf_virtual_channel:2 -> virtual_channel;	/* Virtual channel number. */
    uint32_t _bf_pad:4;
    uint32_t _bf_lane:2 -> lane;	/* Lane number. */
    uint32_t _bf_tid:24 -> tid;
  }
  uint32_t _word1 {
    uint32_t _bf_acq_count:16 -> acq_count;
    uint32_t _bf_op_code:8 -> op_code;
    uint32_t _bf_quad:2 -> quad;	/* Quadrant number. */
  }
  uint32_t _seq_count -> seq_count;
  uint32_t _ticks -> ticks;
  uint32_t _fiducials -> fiducials;
  uint16_t _sbtemp[Nsbtemp] -> sb_temp;
  uint32_t _frame_type -> frame_type;
  int16_t _data[ColumnsPerASIC][ MaxRowsPerASIC*2][ 2] -> data;

  /* Common mode value for a given section, section number can be 0 or 1.
                Will return 0 for data read from XTC, may be non-zero after calibration. */
  float common_mode(uint32_t section)
  [[language("C++")]] @{ return 0; @}
}
} //- @package CsPad2x2
