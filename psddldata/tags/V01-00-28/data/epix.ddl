@include "psddldata/genericpgp.ddl";

@package Epix  {


//------------------ AsicConfigV1 ------------------
@type AsicConfigV1
  [[pack(4)]]
{
  uint32_t _reg1 {
    uint8_t _monostPulser:3 -> monostPulser;
    uint32_t _z:29;
  }
  uint32_t _reg2 {
    uint8_t _dummyTest:1 -> dummyTest;
    uint8_t _dummyMask:1 -> dummyMask;
    uint32_t _z:30;
  }
  uint32_t _reg3 {
    uint16_t _pulser:10 -> pulser;
    uint8_t _pbit:1 -> pbit;
    uint8_t _atest:1 -> atest;
    uint8_t _test:1 -> test;
    uint8_t _sabTest:1 -> sabTest;
    uint8_t _hrTest:1 -> hrTest;
    uint32_t _z:17;
  }
  uint32_t _reg4 {
    uint8_t _digMon1:4 -> digMon1;
    uint8_t _digMon2:4 -> digMon2;
    uint32_t _z:24;
  }
  uint32_t _reg5 {
    uint8_t _pulserDac:3 -> pulserDac;
    uint32_t _z:29;
  }
  uint32_t _reg6 {
    uint8_t _Dm1En:1 -> Dm1En;
    uint8_t _Dm2En:1 -> Dm2En;
    uint8_t _z1:2;
    uint8_t _slvdSBit:1 -> slvdSBit;
    uint32_t _z2:27;
  }
  uint32_t _reg7 {
    uint8_t _VRefDac:6 -> VRefDac;
    uint32_t _z:26;
  }
  uint32_t _reg8 {
    uint8_t _TpsTComp:1 -> TpsTComp;
    uint8_t _TpsMux:4 -> TpsMux;
    uint8_t _RoMonost:3 -> RoMonost;
    uint32_t _z:24;
  }
  uint32_t _reg9 {
    uint8_t _TpsGr:4 -> TpsGr;
    uint8_t _S2dGr:4 -> S2dGr;
    uint32_t _z:24;
  }
  uint32_t _reg10 {
    uint8_t _PpOcbS2d:1 -> PpOcbS2d;
    uint8_t _Ocb:3 -> Ocb;
    uint8_t _Monost:3 -> Monost;
    uint8_t _FastppEnable:1 -> FastppEnable;
    uint32_t _z:24;
  }
  uint32_t _reg11 {
    uint8_t _Preamp:3 -> Preamp;
    uint8_t _z1:1;
    uint8_t _PixelCb:3 -> PixelCb;
    uint32_t _z2:25;
  }
  uint32_t _reg12 {
    uint8_t _S2dTComp:1 -> S2dTComp;
    uint8_t _FilterDac:6 -> FilterDac;
    uint32_t _z:25;
  }
  uint32_t _reg13 {
    uint8_t _TC:2 -> TC;
    uint8_t _S2d:3 -> S2d;
    uint8_t _S2dDacBias:3 -> S2dDacBias;
    uint32_t _z:24;
  }
  uint32_t _reg14 {
    uint8_t _TpsTcDac:2 -> TpsTcDac;
    uint8_t _TpsDac:6 -> TpsDac;
    uint32_t _z:24;
  }
  uint32_t _reg15 {
    uint8_t _S2dTcDac:2 -> S2dTcDac;
    uint8_t _S2dDac:6 -> S2dDac;
    uint32_t _z:24;
  }
  uint32_t _reg16 {
    uint8_t _TestBe:1 -> TestBe;
    uint8_t _IsEn:1 -> IsEn;
    uint8_t _DelExec:1 -> DelExec;
    uint8_t _DelCckReg:1 -> DelCckReg;
    uint32_t _z:28;
  }
  uint32_t _reg17 {
    uint16_t _RowStartAddr:9 -> RowStartAddr;
    uint32_t _z:23;
  }
  uint32_t _reg18 {
    uint16_t _RowStopAddr:9 -> RowStopAddr;
    uint32_t _z:23;
  }
  uint32_t _reg19 {
    uint8_t _ColStartAddr:7 -> ColStartAddr;
    uint32_t _z:25;
  }
  uint32_t _reg20 {
    uint8_t _ColStopAddr:7 -> ColStopAddr;
    uint32_t _z:25;
  }
  uint32_t _reg21 {
    uint16_t _chipID:16 -> chipID;
    uint16_t _z:16;
  }

  /* Constructor with value for each attribute */
  @init()  [[auto]];

}


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_EpixConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  uint32_t _version -> version;
  uint32_t _runTrigDelay -> runTrigDelay;
  uint32_t _daqTrigDelay -> daqTrigDelay;
  uint32_t _dacSetting -> dacSetting;
  uint32_t _asicPins {
    uint8_t _asicGR:1 -> asicGR;
    uint8_t _asicAcq:1 -> asicAcq;
    uint8_t _asicR0:1 -> asicR0;
    uint8_t _asicPpmat:1 -> asicPpmat;
    uint8_t _asicPpbe:1 -> asicPpbe;
    uint8_t _asicRoClk:1 -> asicRoClk;
    uint32_t _z:26;
  }
  uint32_t _asicControls {
    uint8_t _asicGRControl:1 -> asicGRControl;
    uint8_t _asicAcqControl:1 -> asicAcqControl;
    uint8_t _asicR0Control:1 -> asicR0Control;
    uint8_t _asicPpmatControl:1 -> asicPpmatControl;
    uint8_t _asicPpbeControl:1 -> asicPpbeControl;
    uint8_t _asicR0ClkControl:1 -> asicR0ClkControl;
    uint8_t _prepulseR0En:1 -> prepulseR0En;
    uint32_t _adcStreamMode:1 -> adcStreamMode;
    uint8_t _testPatternEnable:1 -> testPatternEnable;
    uint8_t _z1:23;
  }
  uint32_t _acqToAsicR0Delay -> acqToAsicR0Delay;
  uint32_t _asicR0ToAsicAcq -> asicR0ToAsicAcq;
  uint32_t _asicAcqWidth -> asicAcqWidth;
  uint32_t _asicAcqLToPPmatL -> asicAcqLToPPmatL;
  uint32_t _asicRoClkHalfT -> asicRoClkHalfT;
  uint32_t _adcReadsPerPixel -> adcReadsPerPixel;
  uint32_t _adcClkHalfT -> adcClkHalfT;
  uint32_t _asicR0Width -> asicR0Width;
  uint32_t _adcPipelineDelay -> adcPipelineDelay;
  uint32_t _prepulseR0Width -> prepulseR0Width;
  uint32_t _prepulseR0Delay -> prepulseR0Delay;
  uint32_t _digitalCardId0 -> digitalCardId0;
  uint32_t _digitalCardId1 -> digitalCardId1;
  uint32_t _analogCardId0 -> analogCardId0;
  uint32_t _analogCardId1 -> analogCardId1;
  uint32_t _lastRowExclusions -> lastRowExclusions;
  uint32_t _numberOfAsicsPerRow -> numberOfAsicsPerRow;
  uint32_t _numberOfAsicsPerColumn -> numberOfAsicsPerColumn;
  // generally 2 x 2
  uint32_t _numberOfRowsPerAsic -> numberOfRowsPerAsic;
  // for epix100  352
  uint32_t _numberOfPixelsPerAsicRow -> numberOfPixelsPerAsicRow;
  // for epix100 96*4
  uint32_t _baseClockFrequency -> baseClockFrequency;
  uint32_t _asicMask -> asicMask;
  AsicConfigV1 _asics[@self.numberOfAsicsPerRow()*@self.numberOfAsicsPerColumn()] -> asics;
  uint32_t _asicPixelTestArray[@self.numberOfAsicsPerRow()*@self.numberOfAsicsPerColumn()][ @self.numberOfRowsPerAsic()][ (@self.numberOfPixelsPerAsicRow()+31)/32] -> asicPixelTestArray;
  uint32_t _asicPixelMaskArray[@self.numberOfAsicsPerRow()*@self.numberOfAsicsPerColumn()][ @self.numberOfRowsPerAsic()][ (@self.numberOfPixelsPerAsicRow()+31)/32] -> asicPixelMaskArray;

  /* Number of rows in a readout unit */
  uint32_t numberOfRows()  [[inline]]
  [[language("C++")]] @{ return @self.numberOfAsicsPerColumn()*@self.numberOfRowsPerAsic() - @self.lastRowExclusions(); @}

  /* Number of columns in a readout unit */
  uint32_t numberOfColumns()  [[inline]]
  [[language("C++")]] @{ return  @self.numberOfAsicsPerRow()*@self.numberOfPixelsPerAsicRow(); @}

  /* Number of columns in a readout unit */
  uint32_t numberOfAsics()  [[inline]]
  [[language("C++")]] @{ return  @self.numberOfAsicsPerRow()*@self.numberOfAsicsPerColumn(); @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

  /* Constructor which takes values necessary for size calculations */
  @init(numberOfAsicsPerRow -> _numberOfAsicsPerRow, numberOfAsicsPerColumn -> _numberOfAsicsPerColumn, 
      numberOfRowsPerAsic -> _numberOfRowsPerAsic, numberOfPixelsPerAsicRow -> _numberOfPixelsPerAsicRow)  [[inline]];

}

//------------------ Asic10kConfigV1 ------------------
@type Asic10kConfigV1
  [[pack(4)]]
{
  uint32_t _reg1 {
    uint8_t _CompTH_DAC:6 -> CompTH_DAC;
    uint8_t _CompEn_0:1 -> CompEn_0;
    uint8_t _PulserSync:1 -> PulserSync;
    uint32_t _z:24;
  }
  uint32_t _reg2 {
    uint8_t _dummyTest:1 -> dummyTest;
    uint8_t _dummyMask:1 -> dummyMask;
    uint8_t _dummyG:1 -> dummyG;
    uint8_t _dummyGA:1 -> dummyGA;
    uint16_t _dummyUpper12bits:12 -> dummyUpper12bits;
    uint32_t _z:16;
  }
  uint32_t _reg3 {
    uint16_t _pulser:10 -> pulser;
    uint8_t _pbit:1 -> pbit;
    uint8_t _atest:1 -> atest;
    uint8_t _test:1 -> test;
    uint8_t _sabTest:1 -> sabTest;
    uint8_t _hrTest:1 -> hrTest;
    uint8_t _PulserR:1 -> pulserR;
    uint32_t _z:16;
  }
  uint32_t _reg4 {
    uint8_t _digMon1:4 -> digMon1;
    uint8_t _digMon2:4 -> digMon2;
    uint32_t _z:24;
  }
  uint32_t _reg5 {
    uint8_t _pulserDac:3 -> pulserDac;
    uint8_t _monostPulser:3 -> monostPulser;
    uint8_t _CompEn_1:1 -> CompEn_1;
    uint8_t _CompEn_2:1 -> CompEn_2;
    uint32_t _z:24;
  }
  uint32_t _reg6 {
    uint8_t _Dm1En:1 -> Dm1En;
    uint8_t _Dm2En:1 -> Dm2En;
    uint8_t _emph_bd:3 -> emph_bd;
    uint8_t _emph_bc:3 -> emph_bc;
    uint32_t _z:24;
  }
  uint32_t _reg7 {
    uint8_t _VRefDac:6 -> VRefDac;
    uint8_t _VrefLow:2 -> vrefLow;
    uint32_t _z:24;
  }
  uint32_t _reg8 {
    uint8_t _TpsTComp:1 -> TpsTComp;
    uint8_t _TpsMux:4 -> TpsMux;
    uint8_t _RoMonost:3 -> RoMonost;
    uint32_t _z:24;
  }
  uint32_t _reg9 {
    uint8_t _TpsGr:4 -> TpsGr;
    uint8_t _S2dGr:4 -> S2dGr;
    uint32_t _z:24;
  }
  uint32_t _reg10 {
    uint8_t _PpOcbS2d:1 -> PpOcbS2d;
    uint8_t _Ocb:3 -> Ocb;
    uint8_t _Monost:3 -> Monost;
    uint8_t _FastppEnable:1 -> FastppEnable;
    uint32_t _z:24;
  }
  uint32_t _reg11 {
    uint8_t _Preamp:3 -> Preamp;
    uint8_t _PixelCb:3 -> PixelCb;
    uint8_t _Vld1_b:2 -> Vld1_b;
    uint32_t _z:24;
  }
  uint32_t _reg12 {
    uint8_t _S2dTComp:1 -> S2dTComp;
    uint8_t _FilterDac:6 -> FilterDac;
    uint8_t _testVDTransmitter:1 -> testVDTransmitter;
    uint32_t _z:24;
  }
  uint32_t _reg13 {
    uint8_t _TC:2 -> TC;
    uint8_t _S2d:3 -> S2d;
    uint8_t _S2dDacBias:3 -> S2dDacBias;
    uint32_t _z:24;
  }
  uint32_t _reg14 {
    uint8_t _TpsTcDac:2 -> TpsTcDac;
    uint8_t _TpsDac:6 -> TpsDac;
    uint32_t _z:24;
  }
  uint32_t _reg15 {
    uint8_t _S2dTcDac:2 -> S2dTcDac;
    uint8_t _S2dDac:6 -> S2dDac;
    uint32_t _z:24;
  }
  uint32_t _reg16 {
    uint8_t _TestBe:1 -> TestBe;
    uint8_t _IsEn:1 -> IsEn;
    uint8_t _DelExec:1 -> DelExec;
    uint8_t _DelCckReg:1 -> DelCckReg;
    uint8_t _RO_rst_en:1 -> RO_rst_en;
    uint8_t _slvdSBit:1 -> slvdSBit;
    uint8_t _FELmode:1 -> FELmode;
    uint8_t _CompEnOn:1 -> CompEnOn;
    uint32_t _z:24;
  }
  uint32_t _reg17 {
    uint16_t _RowStartAddr:9 -> RowStartAddr;
    uint32_t _z:23;
  }
  uint32_t _reg18 {
    uint16_t _RowStopAddr:9 -> RowStopAddr;
    uint32_t _z:23;
  }
  uint32_t _reg19 {
    uint8_t _ColStartAddr:7 -> ColStartAddr;
    uint32_t _z:25;
  }
  uint32_t _reg20 {
    uint8_t _ColStopAddr:7 -> ColStopAddr;
    uint32_t _z:25;
  }
  uint32_t _reg21 {
    uint16_t _chipID:16 -> chipID;
    uint16_t _z:16;
  }

  /* Constructor with value for each attribute */
  @init()  [[auto]];

}


//------------------ Config10KV1 ------------------
@type Config10KV1
  [[type_id(Id_Epix10kConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  uint32_t _version -> version;
  uint32_t _runTrigDelay -> runTrigDelay;
  uint32_t _daqTrigDelay -> daqTrigDelay;
  uint32_t _dacSetting -> dacSetting;
  uint32_t _asicPins {
    uint8_t _asicGR:1 -> asicGR;
    uint8_t _asicAcq:1 -> asicAcq;
    uint8_t _asicR0:1 -> asicR0;
    uint8_t _asicPpmat:1 -> asicPpmat;
    uint8_t _asicPpbe:1 -> asicPpbe;
    uint8_t _asicRoClk:1 -> asicRoClk;
    uint32_t _z:26;
  }
  uint32_t _asicControls {
    uint8_t _asicGRControl:1 -> asicGRControl;
    uint8_t _asicAcqControl:1 -> asicAcqControl;
    uint8_t _asicR0Control:1 -> asicR0Control;
    uint8_t _asicPpmatControl:1 -> asicPpmatControl;
    uint8_t _asicPpbeControl:1 -> asicPpbeControl;
    uint8_t _asicR0ClkControl:1 -> asicR0ClkControl;
    uint8_t _prepulseR0En:1 -> prepulseR0En;
    uint32_t _adcStreamMode:1 -> adcStreamMode;
    uint8_t _testPatternEnable:1 -> testPatternEnable;
    uint8_t _SyncMode:2 -> SyncMode;  // new
    uint8_t _R0Mode:1 -> R0Mode;  // new
    uint8_t _z1:20;
  }
  uint32_t _DoutPipelineDelay -> DoutPipelineDelay;  // new
  uint32_t _acqToAsicR0Delay -> acqToAsicR0Delay;
  uint32_t _asicR0ToAsicAcq -> asicR0ToAsicAcq;
  uint32_t _asicAcqWidth -> asicAcqWidth;
  uint32_t _asicAcqLToPPmatL -> asicAcqLToPPmatL;
  uint32_t _asicRoClkHalfT -> asicRoClkHalfT;
  uint32_t _adcReadsPerPixel -> adcReadsPerPixel;
  uint32_t _adcClkHalfT -> adcClkHalfT;
  uint32_t _asicR0Width -> asicR0Width;
  uint32_t _adcPipelineDelay -> adcPipelineDelay;
  uint32_t _Sync {  // new
    uint16_t _SyncWidth:16 -> SyncWidth;  // new
    uint16_t _SyncDelay:16 -> SyncDelay;  // new
  }  // new
  uint32_t _prepulseR0Width -> prepulseR0Width;
  uint32_t _prepulseR0Delay -> prepulseR0Delay;
  uint32_t _digitalCardId0 -> digitalCardId0;
  uint32_t _digitalCardId1 -> digitalCardId1;
  uint32_t _analogCardId0 -> analogCardId0;
  uint32_t _analogCardId1 -> analogCardId1;
  uint32_t _lastRowExclusions -> lastRowExclusions;
  uint32_t _numberOfAsicsPerRow -> numberOfAsicsPerRow;
  uint32_t _numberOfAsicsPerColumn -> numberOfAsicsPerColumn;
  uint32_t _numberOfRowsPerAsic -> numberOfRowsPerAsic;
  // for epix10k  176
  uint32_t _numberOfPixelsPerAsicRow -> numberOfPixelsPerAsicRow;
  // for epix10k 48*4
  uint32_t _baseClockFrequency -> baseClockFrequency;
  uint32_t _asicMask -> asicMask;
  uint32_t _Scope {
    uint8_t _scopeEnable:1 -> scopeEnable;
    uint8_t _scopeTrigEdge:1 -> scopeTrigEdge;
    uint8_t _scopeTrigChan:4 -> scopeTrigChan;
    uint8_t _scopeArmMode:2 -> scopeArmMode;
    uint8_t _z:8;
    uint16_t _scopeADCThreshold:16 -> scopeADCThreshold;
  }
  uint32_t _ScopeTriggerParms_1 {
    uint16_t _scopeTrigHoldoff:13 -> scopeTrigHoldoff;
    uint16_t _scopeTrigOffset:13 -> scopeTrigOffset;
  }
  uint32_t _ScopeTriggerParms_2 {
    uint16_t _scopeTraceLength:13 -> scopeTraceLength;
    uint16_t _scopeADCsameplesToSkip:13 -> scopeADCsameplesToSkip;
  }
  uint32_t _ScopeWaveformSelects {
    uint8_t _scopeChanAwaveformSelect:5 -> scopeChanAwaveformSelect;
    uint8_t _scopeChanBwaveformSelect:5 -> scopeChanBwaveformSelect;
    uint32_t _z:22;
  }
  Asic10kConfigV1 _asics[@self.numberOfAsicsPerRow()*@self.numberOfAsicsPerColumn()] -> asics;
  uint16_t _asicPixelConfigArray[@self.numberOfAsicsPerRow()*@self.numberOfAsicsPerColumn()][ @self.numberOfRowsPerAsic()][ (@self.numberOfPixelsPerAsicRow())] -> asicPixelConfigArray;

  /* Number of rows in a readout unit */
  uint32_t numberOfRows()  [[inline]]
  [[language("C++")]] @{ return @self.numberOfAsicsPerColumn()*@self.numberOfRowsPerAsic() - @self.lastRowExclusions(); @}

  /* Number of columns in a readout unit */
  uint32_t numberOfColumns()  [[inline]]
  [[language("C++")]] @{ return  @self.numberOfAsicsPerRow()*@self.numberOfPixelsPerAsicRow(); @}

  /* Number of columns in a readout unit */
  uint32_t numberOfAsics()  [[inline]]
  [[language("C++")]] @{ return  @self.numberOfAsicsPerRow()*@self.numberOfAsicsPerColumn(); @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

  /* Constructor which takes values necessary for size calculations */
  @init(numberOfAsicsPerRow -> _numberOfAsicsPerRow, numberOfAsicsPerColumn -> _numberOfAsicsPerColumn, 
      numberOfRowsPerAsic -> _numberOfRowsPerAsic, numberOfPixelsPerAsicRow -> _numberOfPixelsPerAsicRow,
      lastRowExclusions -> _lastRowExclusions)  [[inline]];

}

//------------------ Asic100aConfigV1 ------------------
@type Asic100aConfigV1
  [[pack(4)]]
{
  uint32_t _reg1 {
    uint8_t _pulserVsPixelOnDelay:3 -> pulserVsPixelOnDelay;
    uint8_t _z1:4;
    uint8_t _pulserSync:1 -> pulserSync;
    uint32_t _z2:24;
  }
  uint32_t _reg2 {
    uint8_t _dummyTest:1 -> dummyTest;
    uint8_t _dummyMask:1 -> dummyMask;
    uint32_t _z:30;
  }
  uint32_t _reg3 {
    uint16_t _testPulserLevel:10 -> testPulserLevel;
    uint8_t _pulserCounterDirection:1     -> pulserCounterDirection;
    uint8_t _automaticTestModeEnable:1    -> automaticTestModeEnable;
    uint8_t _testMode:1     -> testMode;
    uint8_t _testModeWithDarkFrame:1  -> testModeWithDarkFrame;
    uint8_t _highResolutionModeTest:1   -> highResolutionModeTest;
    uint8_t _pulserReset:1  -> pulserReset;
    uint16_t _z:16;
  }
  uint32_t _reg4 {
    uint8_t _digitalMonitorMux1:4 -> digitalMonitorMux1;
    uint8_t _digitalMonitorMux2:4 -> digitalMonitorMux2;
    uint32_t _z:24;
  }
  uint32_t _reg5 {
    uint8_t _testPulserCurrent:3 -> testPulserCurrent;
    uint8_t _z1:1;
    uint8_t _testPointSystemOutputDynamicRange:4 -> testPointSystemOutputDynamicRange;
    uint32_t _z2:24;
  }
  uint32_t _reg6 {
    uint8_t _digitalMonitor1Enable:1 -> digitalMonitor1Enable;
    uint8_t _digitalMonitor2Enable:1 -> digitalMonitor2Enable;
    uint8_t _z1:2;
    uint8_t _LVDS_ImpedenceMatchingEnable:1 -> LVDS_ImpedenceMatchingEnable;
    uint32_t _z2:27;
  }
  uint32_t _reg7 {
    uint8_t _VRefBaseLineDac:6 -> VRefBaselineDac;
    uint8_t _extraRowsLowReferenceValue:2 -> extraRowsLowReferenceValue;
    uint32_t _z:24;
  }
  uint32_t _reg8 {
    uint8_t _testPointSystemTemperatureCompensationEnable:1 -> testPointSystemTemperatureCompensationEnable;
    uint8_t _testPointSytemInputSelect:4 -> testPointSytemInputSelect;
    uint8_t _programmableReadoutDelay:3 -> programmableReadoutDelay;
    uint32_t _z:24;
  }
  uint32_t _reg9 {
    uint8_t _outputDriverOutputDynamicRange0:4 -> outputDriverOutputDynamicRange0;
    uint8_t _outputDriverOutputDynamicRange1:4 -> outputDriverOutputDynamicRange1;
    uint32_t _z:24;
  }
  uint32_t _reg10 {
    uint8_t _balconyEnable:1 -> balconyEnable;
    uint8_t _balconyDriverCurrent:3 -> balconyDriverCurrent;
    uint8_t _fastPowerPulsingSpeed:3 -> fastPowerPulsingSpeed;
    uint8_t _fastPowerPulsingEnable:1 -> fastPowerPulsingEnable;
    uint32_t _z:24;
  }
  uint32_t _reg11 {
    uint8_t _preamplifierCurrent:3 -> preamplifierCurrent;
    uint8_t _pixelOutputBufferCurrent:3 -> pixelOutputBufferCurrent;
    uint8_t _pixelBufferAndPreamplifierDrivingCapabilities:2 -> pixelBufferAndPreamplifierDrivingCapabilities;
    uint32_t _z2:24;
  }
  uint32_t _reg12 {
    uint8_t _outputDriverTemperatureCompensationEnable:1 -> outputDriverTemperatureCompensationEnable;
    uint8_t _pixelFilterLevel:6 -> pixelFilterLevel;
    uint32_t _z:25;
  }
  uint32_t _reg13 {
    uint8_t _bandGapReferenceTemperatureCompensationBits:2 -> bandGapReferenceTemperatureCompensationBits;
    uint8_t _outputDriverDrivingCapabilitiesAndStability:3 -> outputDriverDrivingCapabilitiesAndStability;
    uint8_t _outputDriverDacReferenceBias:3 -> outputDriverDacReferenceBias;
    uint32_t _z:24;
  }
  uint32_t _reg14 {
    uint8_t _testPointSystemTemperatureCompensationGain:2 -> testPointSystemTemperatureCompensationGain;
    uint8_t _testPointSystemInputCommonMode:6 -> testPointSystemInputCommonMode;
    uint32_t _z:24;
  }
  uint32_t _reg15 {
    uint8_t _outputDriverTemperatureCompensationGain0:2 -> outputDriverTemperatureCompensationGain0;
    uint8_t _outputDriverInputCommonMode0:6 -> outputDriverInputCommonMode0;
    uint32_t _z:24;
  }
  uint32_t _reg16 {
    uint8_t _testBackEnd:1 -> testBackEnd;
    uint8_t _interleavedReadOutEnable:1 -> interleavedReadOutEnable;
    uint8_t EXEC_DelayEnable:1 -> EXEC_DelayEnable;
    uint8_t _CCK_RegDelayEnable:1 -> CCK_RegDelayEnable;
    uint8_t _syncPinEnable:1 -> syncPinEnable;
    uint32_t _z:28;
  }
  uint32_t _reg17 {
    uint16_t _RowStartAddr:9 -> RowStartAddr;
    uint32_t _z:23;
  }
  uint32_t _reg18 {
    uint16_t _RowStopAddr:9 -> RowStopAddr;
    uint32_t _z:23;
  }
  uint32_t _reg19 {
    uint8_t _ColumnStartAddr:7 -> ColumnStartAddr;
    uint32_t _z:25;
  }
  uint32_t _reg20 {
    uint8_t _ColumnStopAddr:7 -> ColumnStopAddr;
    uint32_t _z:25;
  }
  uint32_t _reg21 {
    uint16_t _chipID:16 -> chipID;
    uint16_t _z:16;
  }
  uint32_t _reg22 {
    uint8_t _outputDriverOutputDynamicRange2:4 -> outputDriverOutputDynamicRange2;
    uint8_t _outputDriverOutputDynamicRange3:4 -> outputDriverOutputDynamicRange3;
    uint32_t _z:24;   
  }
  uint32_t _reg23 {
    uint8_t _outputDriverTemperatureCompensationGain1:2 -> outputDriverTemperatureCompensationGain1;
    uint8_t _outputDriverInputCommonMode1:6 -> outputDriverInputCommonMode1;
    uint32_t _z:24;
  }
  uint32_t _reg24 {
    uint8_t _outputDriverTemperatureCompensationGain2:2 -> outputDriverTemperatureCompensationGain2;
    uint8_t _outputDriverInputCommonMode2:6 -> outputDriverInputCommonMode2;
    uint32_t _z:24;
  }
  uint32_t _reg25 {
    uint8_t _outputDriverTemperatureCompensationGain3:2 -> outputDriverTemperatureCompensationGain3;
    uint8_t _outputDriverInputCommonMode3:6 -> outputDriverInputCommonMode3;
    uint32_t _z:24;
  }
  

  /* Constructor with value for each attribute */
  @init()  [[auto]];

}

//------------------ Config100aV1 ------------------
@type Config100aV1
  [[type_id(Id_Epix100aConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  uint32_t _version -> version;
  uint32_t _runTrigDelay -> runTrigDelay;
  uint32_t _daqTrigDelay -> daqTrigDelay;
  uint32_t _dacSetting -> dacSetting;
  uint32_t _asicPins {
    uint8_t _asicGR:1 -> asicGR;
    uint8_t _asicAcq:1 -> asicAcq;
    uint8_t _asicR0:1 -> asicR0;
    uint8_t _asicPpmat:1 -> asicPpmat;
    uint8_t _asicPpbe:1 -> asicPpbe;
    uint8_t _asicRoClk:1 -> asicRoClk;
    uint32_t _z:26;
  }
  uint32_t _asicControls {
    uint8_t _asicGRControl:1 -> asicGRControl;
    uint8_t _asicAcqControl:1 -> asicAcqControl;
    uint8_t _asicR0Control:1 -> asicR0Control;
    uint8_t _asicPpmatControl:1 -> asicPpmatControl;
    uint8_t _asicPpbeControl:1 -> asicPpbeControl;
    uint8_t _asicR0ClkControl:1 -> asicR0ClkControl;
    uint8_t _prepulseR0En:1 -> prepulseR0En;
    uint32_t _adcStreamMode:1 -> adcStreamMode;
    uint8_t _testPatternEnable:1 -> testPatternEnable;
    uint8_t _SyncMode:2 -> SyncMode;
    uint8_t _R0Mode:1 -> R0Mode;
    uint8_t _z1:20;
  }
  uint32_t _acqToAsicR0Delay -> acqToAsicR0Delay;
  uint32_t _asicR0ToAsicAcq -> asicR0ToAsicAcq;
  uint32_t _asicAcqWidth -> asicAcqWidth;
  uint32_t _asicAcqLToPPmatL -> asicAcqLToPPmatL;
  uint32_t _asicPPmatToReadout -> asicPPmatToReadout;
  uint32_t _asicRoClkHalfT -> asicRoClkHalfT;
  uint32_t _adcReadsPerPixel -> adcReadsPerPixel;
  uint32_t _adcClkHalfT -> adcClkHalfT;
  uint32_t _asicR0Width -> asicR0Width;
  uint32_t _adcPipelineDelay -> adcPipelineDelay;
  uint32_t _Sync {  // new
    uint16_t _SyncWidth:16 -> SyncWidth;
    uint16_t _SyncDelay:16 -> SyncDelay;
  }  // new
  uint32_t _prepulseR0Width -> prepulseR0Width;
  uint32_t _prepulseR0Delay -> prepulseR0Delay;
  uint32_t _digitalCardId0 -> digitalCardId0;
  uint32_t _digitalCardId1 -> digitalCardId1;
  uint32_t _analogCardId0 -> analogCardId0;
  uint32_t _analogCardId1 -> analogCardId1;
  uint32_t _numberOfAsicsPerRow -> numberOfAsicsPerRow;
  uint32_t _numberOfAsicsPerColumn -> numberOfAsicsPerColumn;
  uint32_t _numberOfRowsPerAsic -> numberOfRowsPerAsic;
  uint32_t _numberOfReadableRowsPerAsic -> numberOfReadableRowsPerAsic;
  // for epix100a  352
  uint32_t _numberOfPixelsPerAsicRow -> numberOfPixelsPerAsicRow;
  // for epix100a 96*4 = 384
  uint32_t _calibrationRowCountPerASIC -> calibrationRowCountPerASIC;
  uint32_t _environmentalRowCountPerASIC -> environmentalRowCountPerASIC;
  uint32_t _baseClockFrequency -> baseClockFrequency;
  uint32_t _asicMask -> asicMask;
  uint32_t _Scope {
    uint8_t _scopeEnable:1 -> scopeEnable;
    uint8_t _scopeTrigEdge:1 -> scopeTrigEdge;
    uint8_t _scopeTrigChan:4 -> scopeTrigChan;
    uint8_t _scopeArmMode:2 -> scopeArmMode;
    uint8_t _z:8;
    uint16_t _scopeADCThreshold:16 -> scopeADCThreshold;
  }
  uint32_t _ScopeTriggerParms_1 {
    uint16_t _scopeTrigHoldoff:13 -> scopeTrigHoldoff;
    uint16_t _scopeTrigOffset:13 -> scopeTrigOffset;
  }
  uint32_t _ScopeTriggerParms_2 {
    uint16_t _scopeTraceLength:13 -> scopeTraceLength;
    uint16_t _scopeADCsameplesToSkip:13 -> scopeADCsameplesToSkip;
  }
  uint32_t _ScopeWaveformSelects {
    uint8_t _scopeChanAwaveformSelect:5 -> scopeChanAwaveformSelect;
    uint8_t _scopeChanBwaveformSelect:5 -> scopeChanBwaveformSelect;
    uint32_t _z:22;
  }
  Asic100aConfigV1 _asics[@self.numberOfAsicsPerRow()*@self.numberOfAsicsPerColumn()] -> asics;
  uint16_t _asicPixelConfigArray[ @self.numberOfRows()][ @self.numberOfColumns()] -> asicPixelConfigArray;
  
  /* Calibration row config map is one row for every two calib rows */
  uint8_t  _calibPixelConfigArray[ @self.numberOfCalibrationRows()  / 2 ][ @self.numberOfPixelsPerAsicRow()*@self.numberOfAsicsPerRow()] -> calibPixelConfigArray;

  /* Number of pixel rows in a readout unit */
  uint32_t numberOfRows()  [[inline]]
  [[language("C++")]] @{ return @self.numberOfAsicsPerColumn()*@self.numberOfRowsPerAsic(); @}

  /* Number of readable pixel rows in a readout unit */
  uint32_t numberOfReadableRows()  [[inline]]
  [[language("C++")]] @{ return @self.numberOfAsicsPerColumn()*@self.numberOfReadableRowsPerAsic(); @}

  /* Number of pixel columns in a readout unit */
  uint32_t numberOfColumns()  [[inline]]
  [[language("C++")]] @{ return  @self.numberOfAsicsPerRow()*@self.numberOfPixelsPerAsicRow(); @}

  /* Number of calibration rows in a readout unit */
  uint32_t numberOfCalibrationRows()  [[inline]]
  [[language("C++")]] @{ return @self.numberOfAsicsPerColumn()*@self.calibrationRowCountPerASIC(); @}

  /* Number of rows in a readout unit */
  uint32_t numberOfEnvironmentalRows()  [[inline]]
  [[language("C++")]] @{ return @self.numberOfAsicsPerColumn()*@self.environmentalRowCountPerASIC(); @}

  /* Number of columns in a readout unit */
  uint32_t numberOfAsics()  [[inline]]
  [[language("C++")]] @{ return  @self.numberOfAsicsPerRow()*@self.numberOfAsicsPerColumn(); @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

  /* Constructor which takes values necessary for size calculations */
  @init(numberOfAsicsPerRow -> _numberOfAsicsPerRow, numberOfAsicsPerColumn -> _numberOfAsicsPerColumn, 
      numberOfRowsPerAsic -> _numberOfRowsPerAsic, numberOfPixelsPerAsicRow -> _numberOfPixelsPerAsicRow,
      calibrationRowCountPerASIC -> _calibrationRowCountPerASIC)  [[inline]];

}

//------------------ AsicSConfigV1 ------------------
@type AsicSConfigV1
  [[pack(4)]]
  [[devel]]
{
  uint32_t _reg1 {
    uint8_t _z1:7;
    uint8_t _pulserSync:1 -> pulserSync;
    uint32_t _z2:24;
  }
  uint32_t _reg2 {
    uint8_t _dummyTest:1 -> dummyTest;
    uint8_t _dummyMask:1 -> dummyMask;
    uint32_t _z:30;
  }
  uint32_t _reg3 {
    uint16_t _testPulserLevel:10 -> testPulserLevel;
    uint8_t _pulserCounterDirection:1     -> pulserCounterDirection;
    uint8_t _automaticTestModeEnable:1    -> automaticTestModeEnable;
    uint8_t _testMode:1     -> testMode;
    uint8_t _testModeWithDarkFrame:1  -> testModeWithDarkFrame;
    uint8_t _highResolutionModeTest:1   -> highResolutionModeTest;
    uint8_t _pulserReset:1  -> pulserReset;
    uint16_t _z:16;
  }
  uint32_t _reg4 {
    uint8_t _digitalMonitorMux1:4 -> digitalMonitorMux1;
    uint8_t _digitalMonitorMux2:4 -> digitalMonitorMux2;
    uint32_t _z:24;
  }
  uint32_t _reg5 {
    uint8_t _testPulserCurrent:3 -> testPulserCurrent;
    uint8_t _pulserVsPixelOnDelay:3 -> pulserVsPixelOnDelay;
    uint32_t _z:26;
  }
  uint32_t _reg6 {
    uint8_t _digitalMonitor1Enable:1 -> digitalMonitor1Enable;
    uint8_t _digitalMonitor2Enable:1 -> digitalMonitor2Enable;
    uint8_t _inputLeakageCompensation:2 -> inputLeakageCompensation;
    uint8_t _inputLeakageCompensationEnable:1 -> inputLeakageCompensationEnable;
    uint32_t _z:27;
  }
  uint32_t _reg7 {
    uint8_t _VRefBaseLineDac:6 -> VRefBaselineDac;
    uint8_t _extraRowsLowReferenceValue:2 -> extraRowsLowReferenceValue;
    uint32_t _z:24;
  }
  uint32_t _reg8 {
    uint8_t _testPointSystemTemperatureCompensationEnable:1 -> testPointSystemTemperatureCompensationEnable;
    uint8_t _testPointSytemInputSelect:4 -> testPointSytemInputSelect;
    uint8_t _programmableReadoutDelay:3 -> programmableReadoutDelay;
    uint32_t _z:24;
  }
  uint32_t _reg9 {
    uint8_t _testPointSystemOutputDynamicRange:4 -> testPointSystemOutputDynamicRange;
    uint8_t _outputDriverOutputDynamicRange:4 -> outputDriverOutputDynamicRange;
    uint32_t _z:24;
  }
  uint32_t _reg10 {
    uint8_t _balconyEnable:1 -> balconyEnable;
    uint8_t _balconyDriverCurrent:3 -> balconyDriverCurrent;
    uint8_t _fastPowerPulsingSpeed:3 -> fastPowerPulsingSpeed;
    uint8_t _fastPowerPulsingEnable:1 -> fastPowerPulsingEnable;
    uint32_t _z:24;
  }
  uint32_t _reg11 {
    uint8_t _preamplifierCurrent:3 -> preamplifierCurrent;
    uint8_t _pixelOutputBufferCurrent:3 -> pixelOutputBufferCurrent;
    uint8_t _pixelBufferAndPreamplifierDrivingCapabilities:2 -> pixelBufferAndPreamplifierDrivingCapabilities;
    uint32_t _z2:24;
  }
  uint32_t _reg12 {
    uint8_t _outputDriverTemperatureCompensationEnable:1 -> outputDriverTemperatureCompensationEnable;
    uint8_t _pixelFilterLevel:6 -> pixelFilterLevel;
    uint32_t _z:25;
  }
  uint32_t _reg13 {
    uint8_t _bandGapReferenceTemperatureCompensationBits:2 -> bandGapReferenceTemperatureCompensationBits;
    uint8_t _outputDriverDrivingCapabilitiesAndStability:3 -> outputDriverDrivingCapabilitiesAndStability;
    uint8_t _outputDriverDacReferenceBias:3 -> outputDriverDacReferenceBias;
    uint32_t _z:24;
  }
  uint32_t _reg14 {
    uint8_t _testPointSystemTemperatureCompensationGain:2 -> testPointSystemTemperatureCompensationGain;
    uint8_t _testPointSystemInputCommonMode:6 -> testPointSystemInputCommonMode;
    uint32_t _z:24;
  }
  uint32_t _reg15 {
    uint8_t _outputDriverTemperatureCompensationGain:2 -> outputDriverTemperatureCompensationGain0;
    uint8_t _outputDriverInputCommonMode:6 -> outputDriverInputCommonMode0;
    uint32_t _z:24;
  }
  uint32_t _reg16 {
    uint8_t _testBackEnd:1 -> testBackEnd;
    uint8_t _interleavedReadOutEnable:1 -> interleavedReadOutEnable;
    uint8_t EXEC_DelayEnable:1 -> EXEC_DelayEnable;
    uint8_t _CCK_RegDelayEnable:1 -> CCK_RegDelayEnable;
    uint8_t _syncPinEnable:1 -> syncPinEnable;
    uint8_t _LVDS_ImpedenceMatchingEnable:1 -> LVDS_ImpedenceMatchingEnable;
    uint32_t _z:26;
  }
  uint32_t _reg17 {
    uint16_t _RowStartAddr:9 -> RowStartAddr;
    uint32_t _z:23;
  }
  uint32_t _reg18 {
    uint16_t _RowStopAddr:9 -> RowStopAddr;
    uint32_t _z:23;
  }
  uint32_t _reg19 {
    uint8_t _ColumnStartAddr:7 -> ColumnStartAddr;
    uint32_t _z:25;
  }
  uint32_t _reg20 {
    uint8_t _ColumnStopAddr:7 -> ColumnStopAddr;
    uint32_t _z:25;
  }
  uint32_t _reg21 {
    uint16_t _chipID:16 -> chipID;
    uint16_t _z:16;
  }
  

  /* Constructor with value for each attribute */
  @init()  [[auto]];

}

//------------------ ConfigSV1 ------------------
@type ConfigSV1
  [[type_id(Id_EpixSConfig, 1)]]
  [[config_type]]
  [[devel]]
  [[pack(4)]]
{
  uint32_t _version -> version;
  uint32_t _runTrigDelay -> runTrigDelay;
  uint32_t _daqTrigDelay -> daqTrigDelay;
  uint32_t _dacSetting -> dacSetting;
  uint32_t _asicPins {
    uint8_t _asicGR:1 -> asicGR;
    uint8_t _asicAcq:1 -> asicAcq;
    uint8_t _asicR0:1 -> asicR0;
    uint8_t _asicPpmat:1 -> asicPpmat;
    uint8_t _asicPpbe:1 -> asicPpbe;
    uint8_t _asicRoClk:1 -> asicRoClk;
    uint32_t _z:26;
  }
  uint32_t _asicControls {
    uint8_t _asicGRControl:1 -> asicGRControl;
    uint8_t _asicAcqControl:1 -> asicAcqControl;
    uint8_t _asicR0Control:1 -> asicR0Control;
    uint8_t _asicPpmatControl:1 -> asicPpmatControl;
    uint8_t _asicPpbeControl:1 -> asicPpbeControl;
    uint8_t _asicR0ClkControl:1 -> asicR0ClkControl;
    uint8_t _prepulseR0En:1 -> prepulseR0En;
    uint32_t _adcStreamMode:1 -> adcStreamMode;
    uint8_t _testPatternEnable:1 -> testPatternEnable;
    uint8_t _SyncMode:2 -> SyncMode;
    uint8_t _R0Mode:1 -> R0Mode;
    uint8_t _z1:20;
  }
  uint32_t _acqToAsicR0Delay -> acqToAsicR0Delay;
  uint32_t _asicR0ToAsicAcq -> asicR0ToAsicAcq;
  uint32_t _asicAcqWidth -> asicAcqWidth;
  uint32_t _asicAcqLToPPmatL -> asicAcqLToPPmatL;
  uint32_t _asicPPmatToReadout -> asicPPmatToReadout;
  uint32_t _asicRoClkHalfT -> asicRoClkHalfT;
  uint32_t _adcReadsPerPixel -> adcReadsPerPixel;
  uint32_t _adcClkHalfT -> adcClkHalfT;
  uint32_t _asicR0Width -> asicR0Width;
  uint32_t _adcPipelineDelay -> adcPipelineDelay;
  uint32_t _Sync {  // new
    uint16_t _SyncWidth:16 -> SyncWidth;
    uint16_t _SyncDelay:16 -> SyncDelay;
  }  // new
  uint32_t _prepulseR0Width -> prepulseR0Width;
  uint32_t _prepulseR0Delay -> prepulseR0Delay;
  uint32_t _digitalCardId0 -> digitalCardId0;
  uint32_t _digitalCardId1 -> digitalCardId1;
  uint32_t _analogCardId0 -> analogCardId0;
  uint32_t _analogCardId1 -> analogCardId1;
  uint32_t _numberOfAsicsPerRow -> numberOfAsicsPerRow;
  uint32_t _numberOfAsicsPerColumn -> numberOfAsicsPerColumn;
  uint32_t _numberOfRowsPerAsic -> numberOfRowsPerAsic;
  uint32_t _numberOfReadableRowsPerAsic -> numberOfReadableRowsPerAsic;
  // for epixS  352
  uint32_t _numberOfPixelsPerAsicRow -> numberOfPixelsPerAsicRow;
  // for epixS 96*4 = 384
  uint32_t _calibrationRowCountPerASIC -> calibrationRowCountPerASIC;
  uint32_t _environmentalRowCountPerASIC -> environmentalRowCountPerASIC;
  uint32_t _baseClockFrequency -> baseClockFrequency;
  uint32_t _asicMask -> asicMask;
  uint32_t _Scope {
    uint8_t _scopeEnable:1 -> scopeEnable;
    uint8_t _scopeTrigEdge:1 -> scopeTrigEdge;
    uint8_t _scopeTrigChan:4 -> scopeTrigChan;
    uint8_t _scopeArmMode:2 -> scopeArmMode;
    uint8_t _z:8;
    uint16_t _scopeADCThreshold:16 -> scopeADCThreshold;
  }
  uint32_t _ScopeTriggerParms_1 {
    uint16_t _scopeTrigHoldoff:13 -> scopeTrigHoldoff;
    uint16_t _scopeTrigOffset:13 -> scopeTrigOffset;
  }
  uint32_t _ScopeTriggerParms_2 {
    uint16_t _scopeTraceLength:13 -> scopeTraceLength;
    uint16_t _scopeADCsameplesToSkip:13 -> scopeADCsameplesToSkip;
  }
  uint32_t _ScopeWaveformSelects {
    uint8_t _scopeChanAwaveformSelect:5 -> scopeChanAwaveformSelect;
    uint8_t _scopeChanBwaveformSelect:5 -> scopeChanBwaveformSelect;
    uint32_t _z:22;
  }
  AsicSConfigV1 _asics[@self.numberOfAsicsPerRow()*@self.numberOfAsicsPerColumn()] -> asics;
  uint16_t _asicPixelConfigArray[ @self.numberOfRows()][ @self.numberOfColumns()] -> asicPixelConfigArray;
  
  /* Calibration row config map is one row for every two calib rows */
  uint8_t  _calibPixelConfigArray[ @self.numberOfCalibrationRows()  / 2 ][ @self.numberOfPixelsPerAsicRow()*@self.numberOfAsicsPerRow()] -> calibPixelConfigArray;

  /* Number of pixel rows in a readout unit */
  uint32_t numberOfRows()  [[inline]]
  [[language("C++")]] @{ return @self.numberOfAsicsPerColumn()*@self.numberOfRowsPerAsic(); @}

  /* Number of readable pixel rows in a readout unit */
  uint32_t numberOfReadableRows()  [[inline]]
  [[language("C++")]] @{ return @self.numberOfAsicsPerColumn()*@self.numberOfReadableRowsPerAsic(); @}

  /* Number of pixel columns in a readout unit */
  uint32_t numberOfColumns()  [[inline]]
  [[language("C++")]] @{ return  @self.numberOfAsicsPerRow()*@self.numberOfPixelsPerAsicRow(); @}

  /* Number of calibration rows in a readout unit */
  uint32_t numberOfCalibrationRows()  [[inline]]
  [[language("C++")]] @{ return @self.numberOfAsicsPerColumn()*@self.calibrationRowCountPerASIC(); @}

  /* Number of rows in a readout unit */
  uint32_t numberOfEnvironmentalRows()  [[inline]]
  [[language("C++")]] @{ return @self.numberOfAsicsPerColumn()*@self.environmentalRowCountPerASIC(); @}

  /* Number of columns in a readout unit */
  uint32_t numberOfAsics()  [[inline]]
  [[language("C++")]] @{ return  @self.numberOfAsicsPerRow()*@self.numberOfAsicsPerColumn(); @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

  /* Constructor which takes values necessary for size calculations */
  @init(numberOfAsicsPerRow -> _numberOfAsicsPerRow, numberOfAsicsPerColumn -> _numberOfAsicsPerColumn, 
      numberOfRowsPerAsic -> _numberOfRowsPerAsic, numberOfPixelsPerAsicRow -> _numberOfPixelsPerAsicRow,
      calibrationRowCountPerASIC -> _calibrationRowCountPerASIC)  [[inline]];

}



//------------------ ElementV1 ------------------
@type ElementV1
  [[type_id(Id_EpixElement, 1)]]
  [[pack(2)]]
  [[config(ConfigV1)]]
  [[config(Config10KV1)]]
  [[config(GenericPgp.ConfigV1)]]
{
  uint32_t _first {
    uint8_t _vc:2 -> vc;
    uint8_t _z:4;
    uint8_t _lane:2 -> lane;
    uint32_t _tid:24;
  }
  uint32_t _second {
    uint16_t _acqCount:16 -> acqCount;
    uint16_t _z:16;
  }
  uint32_t _frameNumber -> frameNumber;
  uint32_t _ticks -> ticks;
  uint32_t _fiducials -> fiducials;
  uint32_t _z0;
  uint32_t _z1;
  uint32_t _z2;
  uint16_t _frame[@config.numberOfRows()][@config.numberOfColumns()] -> frame;
  uint16_t _excludedRows[@config.lastRowExclusions()][@config.numberOfColumns()] -> excludedRows;
  uint16_t _temperatures[@config.numberOfAsics()] -> temperatures;
  uint32_t _lastWord -> lastWord;
}

//------------------ ElementV2 ------------------
@type ElementV2
  [[type_id(Id_EpixElement, 2)]]
  [[pack(2)]]
  [[config(Config100aV1)]]
  [[config(ConfigSV1)]]
{
  uint32_t _first {
    uint8_t _vc:2 -> vc;
    uint8_t _z:4;
    uint8_t _lane:2 -> lane;
    uint32_t _tid:24;
  }
  uint32_t _second {
    uint16_t _acqCount:16 -> acqCount;
    uint16_t _z:16;
  }
  uint32_t _frameNumber -> frameNumber;
  uint32_t _ticks -> ticks;
  uint32_t _fiducials -> fiducials;
  uint32_t _z0;
  uint32_t _z1;
  uint32_t _z2;
  uint16_t _frame[@config.numberOfReadableRows()][@config.numberOfColumns()] -> frame;
  uint16_t _calibrationRows[@config.numberOfCalibrationRows()][@config.numberOfColumns()] -> calibrationRows;
  uint16_t _environmentalRows[@config.numberOfEnvironmentalRows()][@config.numberOfColumns()] -> environmentalRows;
  uint16_t _temperatures[@config.numberOfAsics()] -> temperatures;
  uint32_t _lastWord -> lastWord;
}
} //- @package Epix
