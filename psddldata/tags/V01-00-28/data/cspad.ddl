@package CsPad  {
  /* Defines number of quadrants in a CsPad device. */
  @const int32_t MaxQuadsPerSensor = 4;
  /* Total number of ASICs in one quadrant. */
  @const int32_t ASICsPerQuad = 16;
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
  @const int32_t TwoByTwosPerQuad = 4;
  /* Total number of sectors (2x1) per single quadrant. */
  @const int32_t SectorsPerQuad = 8;
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


//------------------ CsPadDigitalPotsCfg ------------------
/* Class defining configuration for CsPad POTs? */
@type CsPadDigitalPotsCfg
{
  uint8_t _pots[PotsPerQuad] -> pots;

  /* Standard constructore */
  @init()  [[auto]];

}


//------------------ CsPadReadOnlyCfg ------------------
/* Class defining read-only configuration. */
@type CsPadReadOnlyCfg
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


//------------------ CsPadGainMapCfg ------------------
/* Class defining ASIC gain map. */
@type CsPadGainMapCfg
{
  uint16_t _gainMap[ColumnsPerASIC][MaxRowsPerASIC] -> gainMap;	/* Array with the gain map for single ASIC. */

  /* Standard constructor */
  @init()  [[auto]];

}


//------------------ ConfigV1QuadReg ------------------
/* Configuration data for single quadrant. */
@type ConfigV1QuadReg
{
  uint32_t _shiftSelect[TwoByTwosPerQuad] -> shiftSelect;
  uint32_t _edgeSelect[TwoByTwosPerQuad] -> edgeSelect;
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
  CsPadReadOnlyCfg _readOnly -> ro;	/* read-only configuration */
  CsPadDigitalPotsCfg _digitalPots -> dp;
  CsPadGainMapCfg _gainMap -> gm;	/* Gain map. */

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ ConfigV2QuadReg ------------------
/* Configuration data for single quadrant. */
@type ConfigV2QuadReg
{
  uint32_t _shiftSelect[TwoByTwosPerQuad] -> shiftSelect;
  uint32_t _edgeSelect[TwoByTwosPerQuad] -> edgeSelect;
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
  CsPadReadOnlyCfg _readOnly -> ro;	/* read-only configuration */
  CsPadDigitalPotsCfg _digitalPots -> dp;
  CsPadGainMapCfg _gainMap -> gm;	/* Gain map. */

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ ConfigV3QuadReg ------------------
/* Configuration data for single quadrant. */
@type ConfigV3QuadReg
{
  uint32_t _shiftSelect[TwoByTwosPerQuad] -> shiftSelect;
  uint32_t _edgeSelect[TwoByTwosPerQuad] -> edgeSelect;
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
  uint32_t _biasTuning -> biasTuning;
  uint32_t _pdpmndnmBalance -> pdpmndnmBalance;
  CsPadReadOnlyCfg _readOnly -> ro;	/* read-only configuration */
  CsPadDigitalPotsCfg _digitalPots -> dp;
  CsPadGainMapCfg _gainMap -> gm;	/* Gain map. */

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ ConfigV1 ------------------
/* Configuration data for complete CsPad device. */
@type ConfigV1
  [[type_id(Id_CspadConfig, 1)]]
  [[config_type]]
{
  uint32_t _concentratorVersion -> concentratorVersion;
  uint32_t _runDelay -> runDelay;
  uint32_t _eventCode -> eventCode;
  uint32_t _inactiveRunMode -> inactiveRunMode;
  uint32_t _activeRunMode -> activeRunMode;
  uint32_t _testDataIndex -> tdi;
  uint32_t _payloadPerQuad -> payloadSize;
  uint32_t _badAsicMask0 -> badAsicMask0;
  uint32_t _badAsicMask1 -> badAsicMask1;
  uint32_t _AsicMask -> asicMask;
  uint32_t _quadMask -> quadMask;
  ConfigV1QuadReg _quads[MaxQuadsPerSensor] -> quads;

  uint32_t numAsicsRead()
  [[language("C++")]] @{ return (@self.asicMask() & 0xf)==1 ? 4 : 16; @}

  uint32_t numQuads()
  [[language("C++")]] @{ return __builtin_popcount(@self.quadMask()); @}

  uint32_t numSect()
  [[language("C++")]] @{ return @self.numAsicsRead()/2; @}

  /* Constructor with values for scalar attributes */
  @init()  [[auto, inline]];

}


//------------------ ConfigV2 ------------------
/* Configuration data for complete CsPad device. */
@type ConfigV2
  [[type_id(Id_CspadConfig, 2)]]
  [[config_type]]
{
  uint32_t _concentratorVersion -> concentratorVersion;
  uint32_t _runDelay -> runDelay;
  uint32_t _eventCode -> eventCode;
  uint32_t _inactiveRunMode -> inactiveRunMode;
  uint32_t _activeRunMode -> activeRunMode;
  uint32_t _testDataIndex -> tdi;
  uint32_t _payloadPerQuad -> payloadSize;
  uint32_t _badAsicMask0 -> badAsicMask0;
  uint32_t _badAsicMask1 -> badAsicMask1;
  uint32_t _AsicMask -> asicMask;
  uint32_t _quadMask -> quadMask;
  uint32_t _roiMask -> roiMasks;
  ConfigV1QuadReg _quads[MaxQuadsPerSensor] -> quads;

  uint32_t numAsicsRead()
  [[language("C++")]] @{ return (@self.asicMask() & 0xf)==1 ? 4 : 16; @}

  /* ROI mask for given quadrant */
  uint32_t roiMask(uint32_t iq)
  [[language("C++")]] @{ return (@self.roiMasks() >> (8*iq)) & 0xff; @}

  /* Number of ASICs in given quadrant */
  uint32_t numAsicsStored(uint32_t iq)
  [[language("C++")]] @{ return __builtin_popcount(@self.roiMask(iq))*2; @}

  /* Total number of quadrants in setup */
  uint32_t numQuads()
  [[language("C++")]] @{ return __builtin_popcount(@self.quadMask()); @}

  /* Total number of sections (2x1) in all quadrants */
  uint32_t numSect()
  [[language("C++")]] @{ return __builtin_popcount(@self.roiMasks()); @}

  /* Constructor with values for each attributes */
  @init()  [[auto, inline]];

}


//------------------ ConfigV3 ------------------
/* Configuration data for complete CsPad device. */
@type ConfigV3
  [[type_id(Id_CspadConfig, 3)]]
  [[config_type]]
{
  uint32_t _concentratorVersion -> concentratorVersion;
  uint32_t _runDelay -> runDelay;
  uint32_t _eventCode -> eventCode;
  ProtectionSystemThreshold _protectionThresholds[MaxQuadsPerSensor] -> protectionThresholds;
  uint32_t _protectionEnable -> protectionEnable;
  uint32_t _inactiveRunMode -> inactiveRunMode;
  uint32_t _activeRunMode -> activeRunMode;
  uint32_t _testDataIndex -> tdi;
  uint32_t _payloadPerQuad -> payloadSize;
  uint32_t _badAsicMask0 -> badAsicMask0;
  uint32_t _badAsicMask1 -> badAsicMask1;
  uint32_t _AsicMask -> asicMask;
  uint32_t _quadMask -> quadMask;
  uint32_t _roiMask -> roiMasks;
  ConfigV1QuadReg _quads[MaxQuadsPerSensor] -> quads;

  uint32_t numAsicsRead()
  [[language("C++")]] @{ return (@self.asicMask() & 0xf)==1 ? 4 : 16; @}

  /* ROI mask for given quadrant */
  uint32_t roiMask(uint32_t iq)
  [[language("C++")]] @{ return (@self.roiMasks() >> (8*iq)) & 0xff; @}

  /* Number of ASICs in given quadrant */
  uint32_t numAsicsStored(uint32_t iq)
  [[language("C++")]] @{ return __builtin_popcount(@self.roiMask(iq))*2; @}

  /* Total number of quadrants in setup */
  uint32_t numQuads()
  [[language("C++")]] @{ return __builtin_popcount(@self.quadMask()); @}

  /* Total number of sections (2x1) in all quadrants */
  uint32_t numSect()
  [[language("C++")]] @{ return __builtin_popcount(@self.roiMasks()); @}

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ ConfigV4 ------------------
/* Configuration data for complete CsPad device. */
@type ConfigV4
  [[type_id(Id_CspadConfig, 4)]]
  [[config_type]]
{
  uint32_t _concentratorVersion -> concentratorVersion;
  uint32_t _runDelay -> runDelay;
  uint32_t _eventCode -> eventCode;
  ProtectionSystemThreshold _protectionThresholds[MaxQuadsPerSensor] -> protectionThresholds;
  uint32_t _protectionEnable -> protectionEnable;
  uint32_t _inactiveRunMode -> inactiveRunMode;
  uint32_t _activeRunMode -> activeRunMode;
  uint32_t _testDataIndex -> tdi;
  uint32_t _payloadPerQuad -> payloadSize;
  uint32_t _badAsicMask0 -> badAsicMask0;
  uint32_t _badAsicMask1 -> badAsicMask1;
  uint32_t _AsicMask -> asicMask;
  uint32_t _quadMask -> quadMask;
  uint32_t _roiMask -> roiMasks;
  ConfigV2QuadReg _quads[MaxQuadsPerSensor] -> quads;

  uint32_t numAsicsRead()
  [[language("C++")]] @{ return (@self.asicMask() & 0xf)==1 ? 4 : 16; @}

  /* ROI mask for given quadrant */
  uint32_t roiMask(uint32_t iq)
  [[language("C++")]] @{ return (@self.roiMasks() >> (8*iq)) & 0xff; @}

  /* Number of ASICs in given quadrant */
  uint32_t numAsicsStored(uint32_t iq)
  [[language("C++")]] @{ return __builtin_popcount(@self.roiMask(iq))*2; @}

  /* Total number of quadrants in setup */
  uint32_t numQuads()
  [[language("C++")]] @{ return __builtin_popcount(@self.quadMask()); @}

  /* Total number of sections (2x1) in all quadrants */
  uint32_t numSect()
  [[language("C++")]] @{ return __builtin_popcount(@self.roiMasks()); @}

  /* Constructor with values for each attributes */
  @init()  [[auto, inline]];

}


//------------------ ConfigV5 ------------------
/* Configuration data for complete CsPad device. */
@type ConfigV5
  [[type_id(Id_CspadConfig, 5)]]
  [[config_type]]
{
  uint32_t _concentratorVersion -> concentratorVersion;
  uint32_t _runDelay -> runDelay;
  uint32_t _eventCode -> eventCode;
  ProtectionSystemThreshold _protectionThresholds[MaxQuadsPerSensor] -> protectionThresholds;
  uint32_t _protectionEnable -> protectionEnable;
  uint32_t _inactiveRunMode -> inactiveRunMode;
  uint32_t _activeRunMode -> activeRunMode;
  uint32_t _internalTriggerDelay -> internalTriggerDelay;
  uint32_t _testDataIndex -> tdi;
  uint32_t _payloadPerQuad -> payloadSize;
  uint32_t _badAsicMask0 -> badAsicMask0;
  uint32_t _badAsicMask1 -> badAsicMask1;
  uint32_t _AsicMask -> asicMask;
  uint32_t _quadMask -> quadMask;
  uint32_t _roiMask -> roiMasks;
  ConfigV3QuadReg _quads[MaxQuadsPerSensor] -> quads;

  uint32_t numAsicsRead()
  [[language("C++")]] @{ return (@self.asicMask() & 0xf)==1 ? 4 : 16; @}

  /* ROI mask for given quadrant */
  uint32_t roiMask(uint32_t iq)
  [[language("C++")]] @{ return (@self.roiMasks() >> (8*iq)) & 0xff; @}

  /* Number of ASICs in given quadrant */
  uint32_t numAsicsStored(uint32_t iq)
  [[language("C++")]] @{ return __builtin_popcount(@self.roiMask(iq))*2; @}

  /* Total number of quadrants in setup */
  uint32_t numQuads()
  [[language("C++")]] @{ return __builtin_popcount(@self.quadMask()); @}

  /* Total number of sections (2x1) in all quadrants */
  uint32_t numSect()
  [[language("C++")]] @{ return __builtin_popcount(@self.roiMasks()); @}

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}


//------------------ ElementV1 ------------------
/* CsPad data from single CsPad quadrant. */
@type ElementV1
  [[config(ConfigV1, ConfigV2, ConfigV3, ConfigV4, ConfigV5)]]
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
  uint32_t _seq_count -> seq_count;	/* Counter incremented on every event. */
  uint32_t _ticks -> ticks;
  uint32_t _fiducials -> fiducials;
  uint16_t _sbtemp[Nsbtemp] -> sb_temp;
  uint32_t _frame_type -> frame_type;
  int16_t _data[@config.numAsicsRead()/2][ ColumnsPerASIC][ MaxRowsPerASIC*2] -> data;
  uint16_t _extra[2];	/* Unused. */

  /* Returns section mask for this quadrant. Mask can contain up to 8 bits in the lower byte, 
                total bit count gives the number of sections active. */
  uint32_t sectionMask()
  [[language("C++")]] @{ return (@config.asicMask() & 0xf)==1 ? 0x3 : 0xff; @}

  /* Common mode value for a given section, section number can be 0 to config.numAsicsRead()/2.
                Will return 0 for data read from XTC, may be non-zero after calibration. */
  float common_mode(uint32_t section)
  [[language("C++")]] @{ return 0; @}
}


//------------------ DataV1 ------------------
/* CsPad data from whole detector. */
@type DataV1
  [[type_id(Id_CspadElement, 1)]]
  [[config(ConfigV1, ConfigV2, ConfigV3, ConfigV4, ConfigV5)]]
{
  ElementV1 _quads[@config.numQuads()] -> quads;	/* Data objects, one element per quadrant. The size of the array is determined by 
            the numQuads() method of the configuration object. */
}


//------------------ ElementV2 ------------------
/* CsPad data from single CsPad quadrant. */
@type ElementV2
  [[config(ConfigV2, ConfigV3, ConfigV4, ConfigV5)]]
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
  int16_t _data[@config.numAsicsStored(@self.quad())/2][ ColumnsPerASIC][ MaxRowsPerASIC*2] -> data;
  uint16_t _extra[2];	/* Unused. */

  /* Returns section mask for this quadrant. Mask can contain up to 8 bits in the lower byte, 
                total bit count gives the number of sections active. */
  uint32_t sectionMask()
  [[language("C++")]] @{ return @config.roiMask(@self.quad()); @}

  /* Common mode value for a given section, section number can be 0 to config.numSect().
	 Will return 0 for data read from XTC, may be non-zero after calibration. */
  float common_mode(uint32_t section)
  [[language("C++")]] @{ return 0; @}
}


//------------------ DataV2 ------------------
/* CsPad data from whole detector. */
@type DataV2
  [[type_id(Id_CspadElement, 2)]]
  [[no_sizeof]]
  [[config(ConfigV2, ConfigV3, ConfigV4, ConfigV5)]]
{
  ElementV2 _quads[@config.numQuads()] -> quads;	/* Data objects, one element per quadrant. The size of the array is determined by 
            the numQuads() method of the configuration object. */
}
} //- @package CsPad
