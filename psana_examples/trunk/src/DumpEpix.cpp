//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpEpix...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpEpix.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/epix.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpEpix)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpEpix::DumpEpix (const std::string& name)
  : Module(name)
  , m_src()
{
  m_src = configSrc("source", "DetInfo(:Epix)");
  // DetInfo(:Epix) or (:Epix*) does not work for Epix10k and Epix100a, 
  // which need to be specified explicitly in parameters as
  // [psana_examples.DumpEpix]
  // source = DetInfo(:Epix10k)
  // OR
  // source = DetInfo(:Epix100a)
}

//--------------
// Destructor --
//--------------
DumpEpix::~DumpEpix ()
{
}

/// Method which is called at the beginning of the run
void
DumpEpix::beginRun(Event& evt, Env& env)
{
  shared_ptr<Psana::Epix::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1) {
    WithMsgLog(name(), info, str) {
      str << "Psana::Epix::ConfigV1:";
      str << "\n  version = " << config1->version();
      str << "\n  runTrigDelay = " << config1->runTrigDelay();
      str << "\n  daqTrigDelay = " << config1->daqTrigDelay();
      str << "\n  dacSetting = " << config1->dacSetting();
      str << "\n  asicGR = " << int(config1->asicGR());
      str << "\n  asicAcq = " << int(config1->asicAcq());
      str << "\n  asicR0 = " << int(config1->asicR0());
      str << "\n  asicPpmat = " << int(config1->asicPpmat());
      str << "\n  asicPpbe = " << int(config1->asicPpbe());
      str << "\n  asicRoClk = " << int(config1->asicRoClk());
      str << "\n  asicGRControl = " << int(config1->asicGRControl());
      str << "\n  asicAcqControl = " << int(config1->asicAcqControl());
      str << "\n  asicR0Control = " << int(config1->asicR0Control());
      str << "\n  asicPpmatControl = " << int(config1->asicPpmatControl());
      str << "\n  asicPpbeControl = " << int(config1->asicPpbeControl());
      str << "\n  asicR0ClkControl = " << int(config1->asicR0ClkControl());
      str << "\n  prepulseR0En = " << int(config1->prepulseR0En());
      str << "\n  adcStreamMode = " << config1->adcStreamMode();
      str << "\n  testPatternEnable = " << int(config1->testPatternEnable());
      str << "\n  acqToAsicR0Delay = " << config1->acqToAsicR0Delay();
      str << "\n  asicR0ToAsicAcq = " << config1->asicR0ToAsicAcq();
      str << "\n  asicAcqWidth = " << config1->asicAcqWidth();
      str << "\n  asicAcqLToPPmatL = " << config1->asicAcqLToPPmatL();
      str << "\n  asicRoClkHalfT = " << config1->asicRoClkHalfT();
      str << "\n  adcReadsPerPixel = " << config1->adcReadsPerPixel();
      str << "\n  adcClkHalfT = " << config1->adcClkHalfT();
      str << "\n  asicR0Width = " << config1->asicR0Width();
      str << "\n  adcPipelineDelay = " << config1->adcPipelineDelay();
      str << "\n  prepulseR0Width = " << config1->prepulseR0Width();
      str << "\n  prepulseR0Delay = " << config1->prepulseR0Delay();
      str << "\n  digitalCardId0 = " << config1->digitalCardId0();
      str << "\n  digitalCardId1 = " << config1->digitalCardId1();
      str << "\n  analogCardId0 = " << config1->analogCardId0();
      str << "\n  analogCardId1 = " << config1->analogCardId1();
      str << "\n  lastRowExclusions = " << config1->lastRowExclusions();
      str << "\n  numberOfAsicsPerRow = " << config1->numberOfAsicsPerRow();
      str << "\n  numberOfAsicsPerColumn = " << config1->numberOfAsicsPerColumn();
      str << "\n  numberOfRowsPerAsic = " << config1->numberOfRowsPerAsic();
      str << "\n  numberOfPixelsPerAsicRow = " << config1->numberOfPixelsPerAsicRow();
      str << "\n  baseClockFrequency = " << config1->baseClockFrequency();
      str << "\n  asicMask = " << config1->asicMask();
      str << "\n  numberOfRows = " << config1->numberOfRows();
      str << "\n  numberOfColumns = " << config1->numberOfColumns();
      str << "\n  numberOfAsics = " << config1->numberOfAsics();
      str << "\n  asicPixelTestArray = " << config1->asicPixelTestArray();
      str << "\n  asicPixelMaskArray = " << config1->asicPixelMaskArray();

      const int nasics = config1->numberOfAsics();
      for (int i = 0; i != nasics; ++ i) {
        const Psana::Epix::AsicConfigV1& aconfig = config1->asics(i);
        str << "\n    Psana::Epix::AsicConfigV1 #" << i << ":";
        str << "\n      monostPulser = " << int(aconfig.monostPulser());
        str << "\n      dummyTest = " << int(aconfig.dummyTest());
        str << "\n      dummyMask = " << int(aconfig.dummyMask());
        str << "\n      pulser = " << int(aconfig.pulser());
        str << "\n      pbit = " << int(aconfig.pbit());
        str << "\n      atest = " << int(aconfig.atest());
        str << "\n      test = " << int(aconfig.test());
        str << "\n      sabTest = " << int(aconfig.sabTest());
        str << "\n      hrTest = " << int(aconfig.hrTest());
        str << "\n      digMon1 = " << int(aconfig.digMon1());
        str << "\n      digMon2 = " << int(aconfig.digMon2());
        str << "\n      pulserDac = " << int(aconfig.pulserDac());
        str << "\n      Dm1En = " << int(aconfig.Dm1En());
        str << "\n      Dm2En = " << int(aconfig.Dm2En());
        str << "\n      slvdSBit = " << int(aconfig.slvdSBit());
        str << "\n      VRefDac = " << int(aconfig.VRefDac());
        str << "\n      TpsTComp = " << int(aconfig.TpsTComp());
        str << "\n      TpsMux = " << int(aconfig.TpsMux());
        str << "\n      RoMonost = " << int(aconfig.RoMonost());
        str << "\n      TpsGr = " << int(aconfig.TpsGr());
        str << "\n      S2dGr = " << int(aconfig.S2dGr());
        str << "\n      PpOcbS2d = " << int(aconfig.PpOcbS2d());
        str << "\n      Ocb = " << int(aconfig.Ocb());
        str << "\n      Monost = " << int(aconfig.Monost());
        str << "\n      FastppEnable = " << int(aconfig.FastppEnable());
        str << "\n      Preamp = " << int(aconfig.Preamp());
        str << "\n      PixelCb = " << int(aconfig.PixelCb());
        str << "\n      S2dTComp = " << int(aconfig.S2dTComp());
        str << "\n      FilterDac = " << int(aconfig.FilterDac());
        str << "\n      TC = " << int(aconfig.TC());
        str << "\n      S2d = " << int(aconfig.S2d());
        str << "\n      S2dDacBias = " << int(aconfig.S2dDacBias());
        str << "\n      TpsTcDac = " << int(aconfig.TpsTcDac());
        str << "\n      TpsDac = " << int(aconfig.TpsDac());
        str << "\n      S2dTcDac = " << int(aconfig.S2dTcDac());
        str << "\n      S2dDac = " << int(aconfig.S2dDac());
        str << "\n      TestBe = " << int(aconfig.TestBe());
        str << "\n      IsEn = " << int(aconfig.IsEn());
        str << "\n      DelExec = " << int(aconfig.DelExec());
        str << "\n      DelCckReg = " << int(aconfig.DelCckReg());
        str << "\n      RowStartAddr = " << int(aconfig.RowStartAddr());
        str << "\n      RowStopAddr = " << int(aconfig.RowStopAddr());
        str << "\n      ColStartAddr = " << int(aconfig.ColStartAddr());
        str << "\n      ColStopAddr = " << int(aconfig.ColStopAddr());
        str << "\n      chipID = " << int(aconfig.chipID());
      }
    }
  }


  shared_ptr<Psana::Epix::Config10KV1> config10k1 = env.configStore().get(m_src);
  if (config10k1) {
    WithMsgLog(name(), info, str) {
      str << "Psana::Epix::Config10KV1:";
      str << "\n  version                  = " << config10k1->version();
      str << "\n  runTrigDelay             = " << config10k1->runTrigDelay();
      str << "\n  daqTrigDelay             = " << config10k1->daqTrigDelay();
      str << "\n  dacSetting               = " << config10k1->dacSetting();
      str << "\n  asicGR                   = " << int(config10k1->asicGR());
      str << "\n  asicAcq                  = " << int(config10k1->asicAcq());
      str << "\n  asicR0                   = " << int(config10k1->asicR0());
      str << "\n  asicPpmat                = " << int(config10k1->asicPpmat());
      str << "\n  asicPpbe                 = " << int(config10k1->asicPpbe());
      str << "\n  asicRoClk                = " << int(config10k1->asicRoClk());
      str << "\n  asicGRControl            = " << int(config10k1->asicGRControl());
      str << "\n  asicAcqControl           = " << int(config10k1->asicAcqControl());
      str << "\n  asicR0Control            = " << int(config10k1->asicR0Control());
      str << "\n  asicPpmatControl         = " << int(config10k1->asicPpmatControl());
      str << "\n  asicPpbeControl          = " << int(config10k1->asicPpbeControl());
      str << "\n  asicR0ClkControl         = " << int(config10k1->asicR0ClkControl());
      str << "\n  prepulseR0En             = " << int(config10k1->prepulseR0En());
      str << "\n  adcStreamMode            = " << config10k1->adcStreamMode();
      str << "\n  testPatternEnable        = " << int(config10k1->testPatternEnable());
      str << "\n  SyncMode                 = " << int(config10k1->SyncMode());
      str << "\n  R0Mode                   = " << int(config10k1->R0Mode());
      str << "\n  DoutPipelineDelay        = " << int(config10k1->DoutPipelineDelay());
      str << "\n  acqToAsicR0Delay         = " << int(config10k1->acqToAsicR0Delay());
      str << "\n  asicR0ToAsicAcq          = " << int(config10k1->asicR0ToAsicAcq());
      str << "\n  asicAcqWidth             = " << int(config10k1->asicAcqWidth());
      str << "\n  asicAcqLToPPmatL         = " << int(config10k1->asicAcqLToPPmatL());
      str << "\n  asicRoClkHalfT           = " << int(config10k1->asicRoClkHalfT());
      str << "\n  adcReadsPerPixel         = " << int(config10k1->adcReadsPerPixel());
      str << "\n  adcClkHalfT              = " << int(config10k1->adcClkHalfT());
      str << "\n  asicR0Width              = " << int(config10k1->asicR0Width());
      str << "\n  adcPipelineDelay         = " << int(config10k1->adcPipelineDelay());
      str << "\n  SyncWidth()              = " << int(config10k1->SyncWidth());
      str << "\n  SyncDelay()              = " << int(config10k1->SyncDelay());
      str << "\n  prepulseR0Width          = " << config10k1->prepulseR0Width();
      str << "\n  prepulseR0Delay          = " << config10k1->prepulseR0Delay();
      str << "\n  digitalCardId0           = " << config10k1->digitalCardId0();
      str << "\n  digitalCardId1           = " << config10k1->digitalCardId1();
      str << "\n  analogCardId0            = " << config10k1->analogCardId0();
      str << "\n  analogCardId1            = " << config10k1->analogCardId1();
      str << "\n  lastRowExclusions        = " << config10k1->lastRowExclusions();
      str << "\n  numberOfAsicsPerRow      = " << config10k1->numberOfAsicsPerRow();
      str << "\n  numberOfAsicsPerColumn   = " << config10k1->numberOfAsicsPerColumn();
      str << "\n  numberOfRowsPerAsic      = " << config10k1->numberOfRowsPerAsic();
      str << "\n  numberOfPixelsPerAsicRow = " << config10k1->numberOfPixelsPerAsicRow();
      str << "\n  baseClockFrequency       = " << config10k1->baseClockFrequency();
      str << "\n  asicMask                 = " << config10k1->asicMask();
      str << "\n  scopeEnable              = " << int(config10k1->scopeEnable());
      str << "\n  scopeTrigEdge            = " << int(config10k1->scopeTrigEdge());
      str << "\n  scopeTrigChan            = " << int(config10k1->scopeTrigChan());
      str << "\n  scopeArmMode             = " << int(config10k1->scopeArmMode());
      str << "\n  scopeADCThreshold        = " << int(config10k1->scopeADCThreshold());
      str << "\n  scopeTrigHoldoff         = " << int(config10k1->scopeTrigHoldoff());
      str << "\n  scopeTrigOffset          = " << int(config10k1->scopeTrigOffset());
      str << "\n  scopeTraceLength         = " << int(config10k1->scopeTraceLength());
      str << "\n  scopeADCsameplesToSkip   = " << int(config10k1->scopeADCsameplesToSkip());
      str << "\n  scopeChanAwaveformSelect = " << int(config10k1->scopeChanAwaveformSelect());
      str << "\n  scopeChanBwaveformSelect = " << int(config10k1->scopeChanBwaveformSelect());
      str << "\n  asicPixelConfigArray     = " << config10k1->asicPixelConfigArray();
      str << "\n  numberOfRows             = " << config10k1->numberOfRows();
      str << "\n  numberOfColumns          = " << config10k1->numberOfColumns();
      str << "\n  numberOfAsics            = " << config10k1->numberOfAsics();

      //virtual std::vector<int> asics_shape() const = 0;
      str << "\n  asics_shape()            =";
      std::vector<int> v = config10k1->asics_shape();
      for(std::vector<int>::iterator it = v.begin(); it != v.end(); ++it) { str << " " << *it; }

      const int nasics = config10k1->numberOfAsics();
      for (int i = 0; i != nasics; ++ i) {
        const Psana::Epix::Asic10kConfigV1& aconfig = config10k1->asics(i);
        str << "\n    Psana::Epix::Asic10kConfigV1 #" << i << ":";

        str << "\n      CompTH_DAC       = " << int(aconfig.CompTH_DAC());
        str << "\n      CompEn_0         = " << int(aconfig.CompEn_0());
        str << "\n      PulserSync       = " << int(aconfig.PulserSync());
        str << "\n      dummyTest        = " << int(aconfig.dummyTest());
        str << "\n      dummyMask        = " << int(aconfig.dummyMask());
        str << "\n      dummyG           = " << int(aconfig.dummyG());
        str << "\n      dummyGA          = " << int(aconfig.dummyGA());
        str << "\n      dummyUpper12bits = " << int(aconfig.dummyUpper12bits());
        str << "\n      pulser           = " << int(aconfig.pulser());
        str << "\n      pbit             = " << int(aconfig.pbit());
        str << "\n      atest            = " << int(aconfig.atest());
        str << "\n      test             = " << int(aconfig.test());
        str << "\n      sabTest          = " << int(aconfig.sabTest());
        str << "\n      hrTest           = " << int(aconfig.hrTest());
        str << "\n      pulserR          = " << int(aconfig.pulserR());
        str << "\n      digMon1          = " << int(aconfig.digMon1());
        str << "\n      digMon2          = " << int(aconfig.digMon2());
        str << "\n      pulserDac        = " << int(aconfig.pulserDac());
        str << "\n      monostPulser     = " << int(aconfig.monostPulser());
        str << "\n      CompEn_1         = " << int(aconfig.CompEn_1());
        str << "\n      CompEn_2         = " << int(aconfig.CompEn_2());
        str << "\n      Dm1En            = " << int(aconfig.Dm1En());
        str << "\n      Dm2En            = " << int(aconfig.Dm2En());
        str << "\n      emph_bd          = " << int(aconfig.emph_bd());
        str << "\n      emph_bc          = " << int(aconfig.emph_bc());
        str << "\n      VRefDac          = " << int(aconfig.VRefDac());
        str << "\n      vrefLow          = " << int(aconfig.vrefLow());
        str << "\n      TpsTComp         = " << int(aconfig.TpsTComp());
        str << "\n      TpsMux           = " << int(aconfig.TpsMux());
        str << "\n      RoMonost         = " << int(aconfig.RoMonost());
        str << "\n      TpsGr            = " << int(aconfig.TpsGr());
        str << "\n      S2dGr            = " << int(aconfig.S2dGr());
        str << "\n      PpOcbS2d         = " << int(aconfig.PpOcbS2d());
        str << "\n      Ocb              = " << int(aconfig.Ocb());
        str << "\n      Monost           = " << int(aconfig.Monost());
        str << "\n      FastppEnable     = " << int(aconfig.FastppEnable());
        str << "\n      Preamp           = " << int(aconfig.Preamp());
        str << "\n      PixelCb          = " << int(aconfig.PixelCb());
        str << "\n      Vld1_b           = " << int(aconfig.Vld1_b());
        str << "\n      S2dTComp         = " << int(aconfig.S2dTComp());
        str << "\n      FilterDac        = " << int(aconfig.FilterDac());
        str << "\n      testVDTransmitter= " << int(aconfig.testVDTransmitter());
        str << "\n      TC               = " << int(aconfig.TC());
        str << "\n      S2d              = " << int(aconfig.S2d());
        str << "\n      S2dDacBias       = " << int(aconfig.S2dDacBias());
        str << "\n      TpsTcDac         = " << int(aconfig.TpsTcDac());
        str << "\n      TpsDac           = " << int(aconfig.TpsDac());
        str << "\n      S2dTcDac         = " << int(aconfig.S2dTcDac());
        str << "\n      S2dDac           = " << int(aconfig.S2dDac());
        str << "\n      TestBe           = " << int(aconfig.TestBe());
        str << "\n      IsEn             = " << int(aconfig.IsEn());
        str << "\n      DelExec          = " << int(aconfig.DelExec());
        str << "\n      DelCckReg        = " << int(aconfig.DelCckReg());
        str << "\n      RO_rst_en        = " << int(aconfig.RO_rst_en());
        str << "\n      slvdSBit         = " << int(aconfig.slvdSBit());
        str << "\n      FELmode          = " << int(aconfig.FELmode());
        str << "\n      CompEnOn         = " << int(aconfig.CompEnOn());
        str << "\n      RowStartAddr     = " << int(aconfig.RowStartAddr());
        str << "\n      RowStopAddr      = " << int(aconfig.RowStopAddr());
        str << "\n      ColStartAddr     = " << int(aconfig.ColStartAddr());
        str << "\n      ColStopAddr      = " << int(aconfig.ColStopAddr());
        str << "\n      chipID           = " << int(aconfig.chipID());
      }
    }
  }


  shared_ptr<Psana::Epix::Config100aV1> config100a = env.configStore().get(m_src);
  if (config100a) {
    WithMsgLog(name(), info, str) {
      str << "Psana::Epix::Config100aV1:";
      str << "\n  version                     = " << config100a->version();
      str << "\n  runTrigDelay                = " << config100a->runTrigDelay();
      str << "\n  daqTrigDelay                = " << config100a->daqTrigDelay();
      str << "\n  dacSetting                  = " << config100a->dacSetting();
      str << "\n  asicGR                      = " << int(config100a->asicGR());
      str << "\n  asicAcq                     = " << int(config100a->asicAcq());
      str << "\n  asicR0                      = " << int(config100a->asicR0());
      str << "\n  asicPpmat                   = " << int(config100a->asicPpmat());
      str << "\n  asicPpbe                    = " << int(config100a->asicPpbe());
      str << "\n  asicRoClk                   = " << int(config100a->asicRoClk());
      str << "\n  asicGRControl               = " << int(config100a->asicGRControl());
      str << "\n  asicAcqControl              = " << int(config100a->asicAcqControl());
      str << "\n  asicR0Control               = " << int(config100a->asicR0Control());
      str << "\n  asicPpmatControl            = " << int(config100a->asicPpmatControl());
      str << "\n  asicPpbeControl             = " << int(config100a->asicPpbeControl());
      str << "\n  asicR0ClkControl            = " << int(config100a->asicR0ClkControl());
      str << "\n  prepulseR0En                = " << int(config100a->prepulseR0En());
      str << "\n  adcStreamMode               = " << int(config100a->adcStreamMode());
      str << "\n  testPatternEnable           = " << int(config100a->testPatternEnable());
      str << "\n  SyncMode                    = " << int(config100a->SyncMode());
      str << "\n  R0Mode                      = " << int(config100a->R0Mode());
      str << "\n  acqToAsicR0Delay            = " << int(config100a->acqToAsicR0Delay());
      str << "\n  asicR0ToAsicAcq             = " << int(config100a->asicR0ToAsicAcq());
      str << "\n  asicAcqWidth                = " << int(config100a->asicAcqWidth());
      str << "\n  asicAcqLToPPmatL            = " << int(config100a->asicAcqLToPPmatL());
      str << "\n  asicPPmatToReadout          = " << int(config100a->asicPPmatToReadout());
      str << "\n  asicRoClkHalfT              = " << int(config100a->asicRoClkHalfT());
      str << "\n  adcReadsPerPixel            = " << int(config100a->adcReadsPerPixel());
      str << "\n  adcClkHalfT                 = " << int(config100a->adcClkHalfT());
      str << "\n  asicR0Width                 = " << int(config100a->asicR0Width());
      str << "\n  adcPipelineDelay            = " << int(config100a->adcPipelineDelay());
      str << "\n  SyncWidth                   = " << int(config100a->SyncWidth());
      str << "\n  SyncDelay                   = " << int(config100a->SyncDelay());
      str << "\n  prepulseR0Width             = " << int(config100a->prepulseR0Width());
      str << "\n  prepulseR0Delay             = " << int(config100a->prepulseR0Delay());
      str << "\n  digitalCardId0              = " << int(config100a->digitalCardId0());
      str << "\n  digitalCardId1              = " << int(config100a->digitalCardId1());
      str << "\n  analogCardId0               = " << int(config100a->analogCardId0());
      str << "\n  analogCardId1               = " << int(config100a->analogCardId1());
      str << "\n  numberOfAsicsPerRow         = " << int(config100a->numberOfAsicsPerRow());
      str << "\n  numberOfAsicsPerColumn      = " << int(config100a->numberOfAsicsPerColumn());
      str << "\n  numberOfRowsPerAsic         = " << int(config100a->numberOfRowsPerAsic());
      str << "\n  numberOfReadableRowsPerAsic = " << int(config100a->numberOfReadableRowsPerAsic());
      str << "\n  numberOfPixelsPerAsicRow    = " << int(config100a->numberOfPixelsPerAsicRow());
      str << "\n  calibrationRowCountPerASIC  = " << int(config100a->calibrationRowCountPerASIC());
      str << "\n  environmentalRowCountPerASIC= " << int(config100a->environmentalRowCountPerASIC());
      str << "\n  baseClockFrequency          = " << int(config100a->baseClockFrequency());
      str << "\n  asicMask                    = " << int(config100a->asicMask());
      str << "\n  scopeEnable                 = " << int(config100a->scopeEnable());
      str << "\n  scopeTrigEdge               = " << int(config100a->scopeTrigEdge());
      str << "\n  scopeTrigChan               = " << int(config100a->scopeTrigChan());
      str << "\n  scopeArmMode                = " << int(config100a->scopeArmMode());
      str << "\n  scopeADCThreshold           = " << int(config100a->scopeADCThreshold());
      str << "\n  scopeTrigHoldoff            = " << int(config100a->scopeTrigHoldoff());
      str << "\n  scopeTrigOffset             = " << int(config100a->scopeTrigOffset());
      str << "\n  scopeTraceLength            = " << int(config100a->scopeTraceLength());
      str << "\n  scopeADCsameplesToSkip      = " << int(config100a->scopeADCsameplesToSkip());
      str << "\n  scopeChanAwaveformSelect    = " << int(config100a->scopeChanAwaveformSelect());
      str << "\n  scopeChanBwaveformSelect    = " << int(config100a->scopeChanBwaveformSelect());
      str << "\n  numberOfRows                = " << int(config100a->numberOfRows());
      str << "\n  numberOfReadableRows        = " << int(config100a->numberOfReadableRows());
      str << "\n  numberOfColumns             = " << int(config100a->numberOfColumns());
      str << "\n  numberOfCalibrationRows     = " << int(config100a->numberOfCalibrationRows());
      str << "\n  numberOfEnvironmentalRows   = " << int(config100a->numberOfEnvironmentalRows());
      str << "\n  numberOfAsics               = " << int(config100a->numberOfAsics());

      //virtual ndarray<const uint16_t, 2> asicPixelConfigArray() const = 0;
      str << "\n  asicPixelConfigArray() = " << config100a->asicPixelConfigArray();    //New

      //virtual ndarray<const uint8_t, 2> calibPixelConfigArray() const = 0;
      str << "\n  calibPixelConfigArray() =" << config100a->calibPixelConfigArray();    //New;
      //str << "\n  calibPixelConfigArray() =";    //New;
      //ndarray<const uint8_t, 2>& nda = config100a->calibPixelConfigArray();
      //for(std::vector<int>::iterator it = v.begin(); it != v.end(); ++it) { str << " " << *it; }




      //virtual std::vector<int> asics_shape() const = 0;
      str << "\n  asics_shape()            =";
      std::vector<int> v = config100a->asics_shape();
      for(std::vector<int>::iterator it = v.begin(); it != v.end(); ++it) { str << " " << *it; }

      //virtual const Epix::Asic100aConfigV1& asics(uint32_t i0) const = 0;
      const int nasics = config100a->numberOfAsics();
      for (int i = 0; i != nasics; ++ i) {
        const Psana::Epix::Asic100aConfigV1& aconfig = config100a->asics(i);
        str << "\n   Psana::Epix::Asic100aConfigV1 #" << i << ":";
        str << "\n     pulserVsPixelOnDelay                          = " << int(aconfig.pulserVsPixelOnDelay());
        str << "\n     pulserSync                                    = " << int(aconfig.pulserSync());
        str << "\n     dummyTest                                     = " << int(aconfig.dummyTest());
        str << "\n     dummyMask                                     = " << int(aconfig.dummyMask());
        str << "\n     testPulserLevel                               = " << int(aconfig.testPulserLevel());
        str << "\n     pulserCounterDirection                        = " << int(aconfig.pulserCounterDirection());
        str << "\n     automaticTestModeEnable                       = " << int(aconfig.automaticTestModeEnable());
        str << "\n     testMode                                      = " << int(aconfig.testMode());
        str << "\n     testModeWithDarkFrame                         = " << int(aconfig.testModeWithDarkFrame());
        str << "\n     highResolutionModeTest                        = " << int(aconfig.highResolutionModeTest());
        str << "\n     pulserReset                                   = " << int(aconfig.pulserReset());
        str << "\n     digitalMonitorMux1                            = " << int(aconfig.digitalMonitorMux1());
        str << "\n     digitalMonitorMux2                            = " << int(aconfig.digitalMonitorMux2());
        str << "\n     testPulserCurrent                             = " << int(aconfig.testPulserCurrent());
        str << "\n     testPointSystemOutputDynamicRange             = " << int(aconfig.testPointSystemOutputDynamicRange());
        str << "\n     digitalMonitor1Enable                         = " << int(aconfig.digitalMonitor1Enable());
        str << "\n     digitalMonitor2Enable                         = " << int(aconfig.digitalMonitor2Enable());
        str << "\n     LVDS_ImpedenceMatchingEnable                  = " << int(aconfig.LVDS_ImpedenceMatchingEnable());
        str << "\n     VRefBaselineDac                               = " << int(aconfig.VRefBaselineDac());
        str << "\n     extraRowsLowReferenceValue                    = " << int(aconfig.extraRowsLowReferenceValue());
        str << "\n     testPointSystemTemperatureCompensationEnable  = " << int(aconfig.testPointSystemTemperatureCompensationEnable()  );
        str << "\n     testPointSytemInputSelect                     = " << int(aconfig.testPointSytemInputSelect());
        str << "\n     programmableReadoutDelay                      = " << int(aconfig.programmableReadoutDelay());
        str << "\n     outputDriverOutputDynamicRange0               = " << int(aconfig.outputDriverOutputDynamicRange0());
        str << "\n     outputDriverOutputDynamicRange1               = " << int(aconfig.outputDriverOutputDynamicRange1());
        str << "\n     balconyEnable                                 = " << int(aconfig.balconyEnable());
        str << "\n     balconyDriverCurrent                          = " << int(aconfig.balconyDriverCurrent());
        str << "\n     fastPowerPulsingSpeed                         = " << int(aconfig.fastPowerPulsingSpeed());
        str << "\n     fastPowerPulsingEnable                        = " << int(aconfig.fastPowerPulsingEnable());
        str << "\n     preamplifierCurrent                           = " << int(aconfig.preamplifierCurrent());
        str << "\n     pixelOutputBufferCurrent                      = " << int(aconfig.pixelOutputBufferCurrent());
        str << "\n     pixelBufferAndPreamplifierDrivingCapabilities = " << int(aconfig.pixelBufferAndPreamplifierDrivingCapabilities());
        str << "\n     outputDriverTemperatureCompensationEnable     = " << int(aconfig.outputDriverTemperatureCompensationEnable());
        str << "\n     pixelFilterLevel                              = " << int(aconfig.pixelFilterLevel());
        str << "\n     bandGapReferenceTemperatureCompensationBits   = " << int(aconfig.bandGapReferenceTemperatureCompensationBits());
        str << "\n     outputDriverDrivingCapabilitiesAndStability   = " << int(aconfig.outputDriverDrivingCapabilitiesAndStability());
        str << "\n     outputDriverDacReferenceBias                  = " << int(aconfig.outputDriverDacReferenceBias());
        str << "\n     testPointSystemTemperatureCompensationGain    = " << int(aconfig.testPointSystemTemperatureCompensationGain());
        str << "\n     testPointSystemInputCommonMode                = " << int(aconfig.testPointSystemInputCommonMode());
        str << "\n     outputDriverTemperatureCompensationGain0      = " << int(aconfig.outputDriverTemperatureCompensationGain0());
        str << "\n     outputDriverInputCommonMode0                  = " << int(aconfig.outputDriverInputCommonMode0());
        str << "\n     testBackEnd                                   = " << int(aconfig.testBackEnd());
        str << "\n     interleavedReadOutEnable                      = " << int(aconfig.interleavedReadOutEnable());
        str << "\n     EXEC_DelayEnable                              = " << int(aconfig.EXEC_DelayEnable());
        str << "\n     CCK_RegDelayEnable                            = " << int(aconfig.CCK_RegDelayEnable());
        str << "\n     syncPinEnable                                 = " << int(aconfig.syncPinEnable());
        str << "\n     RowStartAddr                                  = " << int(aconfig.RowStartAddr());
        str << "\n     RowStopAddr                                   = " << int(aconfig.RowStopAddr());
        str << "\n     ColumnStartAddr                               = " << int(aconfig.ColumnStartAddr());
      }
    }
  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpEpix::event(Event& evt, Env& env)
{
  Pds::Src actualSrc;
  shared_ptr<Psana::Epix::ElementV1> data1 = evt.get(m_src, "", &actualSrc);
  if (data1) {
    WithMsgLog(name(), info, str) {
      str << "Epix::ElementV1 at source " << actualSrc;
      str << "\n  vc = " << int(data1->vc());
      str << "\n  lane = " << int(data1->lane());
      str << "\n  acqCount = " << data1->acqCount();
      str << "\n  frameNumber = " << data1->frameNumber();
      str << "\n  ticks = " << data1->ticks();
      str << "\n  fiducials = " << data1->fiducials();
      str << "\n  frame = " << data1->frame();
      str << "\n  excludedRows = " << data1->excludedRows();
      str << "\n  temperatures = " << data1->temperatures();
      str << "\n  lastWord = " << data1->lastWord();
    }
  }

  shared_ptr<Psana::Epix::ElementV2> data2 = evt.get(m_src, "", &actualSrc);
  if (data2) {
    WithMsgLog(name(), info, str) {
      str << "Epix::ElementV2 at source " << actualSrc;
      str << "\n  vc = " << int(data2->vc());
      str << "\n  lane = " << int(data2->lane());
      str << "\n  acqCount = " << data2->acqCount();
      str << "\n  frameNumber = " << data2->frameNumber();
      str << "\n  ticks = " << data2->ticks();
      str << "\n  fiducials = " << data2->fiducials();
      str << "\n  frame = " << data2->frame();
      str << "\n  calibrationRows = " << data2->calibrationRows();        //New
      str << "\n  environmentalRows = " << data2->environmentalRows();    //New
      str << "\n  temperatures = " << data2->temperatures();
      str << "\n  lastWord = " << data2->lastWord();
    }
  }
}

} // namespace psana_examples
