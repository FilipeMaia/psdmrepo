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
    uint32_t _z:29;
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
    uint32_t _z:25;
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
  uint32_t _numberOfRowsPerAsic -> numberOfRowsPerAsic;
  uint32_t _numberOfPixelsPerAsicRow -> numberOfPixelsPerAsicRow;
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


//------------------ ElementV1 ------------------
@type ElementV1
  [[type_id(Id_EpixElement, 1)]]
  [[pack(2)]]
  [[config(ConfigV1)]]
{
  uint32_t _first {
    uint8_t _vc:2 -> vc;
    uint8_t _z:4;
    uint8_t _lane:2 -> lane;
    uint32_t _tid:24;
  }
  uint32_t _second {
    uint16_t _acqCount:16 -> acqCount;
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
} //- @package Epix
