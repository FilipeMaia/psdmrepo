//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadConfigPars...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "CSPadPixCoords/CSPadConfigPars.h"

//-----------------
// C/C++ Headers --
//-----------------
//#include <time.h>

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace CSPadPixCoords;

using namespace std;

namespace CSPadPixCoords {

//----------------
// Constructors --
//----------------

CSPadConfigPars::CSPadConfigPars ()
{
  m_source = PSEvt::Source("DetInfo(:Cspad)");
  setCSPadConfigParsDefault();
}

CSPadConfigPars::CSPadConfigPars (PSEvt::Source source)
    : m_source(source)
{
  setCSPadConfigParsDefault();
}

CSPadConfigPars::CSPadConfigPars ( uint32_t numQuads,
				   uint32_t quadNumber[],
				   uint32_t roiMask[]
				  )
{
  m_numQuads = numQuads;
  m_num2x1StoredInData = 0;

  for (uint32_t q = 0; q < m_numQuads; ++ q) {
    m_quadNumber[q]       = quadNumber[q];
    m_roiMask[q]          = roiMask[q];
    m_num2x1Stored[q]     = getNum2x1InMask (roiMask[q]);
    m_num2x1StoredInData += m_num2x1Stored[q]; 
  }
}

//--------------
// Destructor --
//--------------

CSPadConfigPars::~CSPadConfigPars ()
{
}

//--------------------

uint32_t 
CSPadConfigPars::getNum2x1InMask(uint32_t mask)
{
  uint32_t num2x1InMask=0;
  for(uint32_t sect=0; sect < 8; sect++) {
    if( mask & (1<<sect) ) num2x1InMask++; 
  }
  return num2x1InMask;
}

//--------------------

void 
CSPadConfigPars::setCSPadConfigParsDefault()
{
  m_numQuads = NQuadsMax;
  m_num2x1StoredInData = 0;

  for (uint32_t q = 0; q < m_numQuads; ++ q) {
    m_quadNumber[q]       = q;
    m_roiMask[q]          = 0377; // or 255;
    m_num2x1Stored[q]     = 8;
    m_num2x1StoredInData += m_num2x1Stored[q]; 
  }
  m_config_vers    = "N/A yet";
  m_data_vers      = "N/A yet";
}

//--------------------

/// Print input parameters
void 
CSPadConfigPars::printCSPadConfigPars()
{
  WithMsgLog(name(), info, log) {
    log << "CSPAD config pars from "      << m_config_vers
        << " and "                        << m_data_vers
        << "\n  N configs found:"         << m_count_cfg
        << "\n  number of quads stored: " << m_numQuads    
        << "\n  number of 2x1 stored:   " << m_num2x1StoredInData;    

    for (uint32_t q = 0; q < m_numQuads; ++ q) {
      log << "\n  quad="  << m_quadNumber[q]
          << "  roiMask=" << m_roiMask[q]
          << "  num2x1="  << m_num2x1Stored[q];
    }
      log << "\n";
  }
}

//--------------------

void 
CSPadConfigPars::setCSPadConfigPars(PSEvt::Event& evt, PSEnv::Env& env)
{
  setCSPadConfigParsFromEnv(env);
  setCSPadConfigParsFromEvent(evt);
}

//--------------------

void 
CSPadConfigPars::setCSPadConfigParsFromEnv(PSEnv::Env& env)
{
  m_count_cfg = 0;
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV2>(env) ) { m_config_vers = "CsPad::ConfigV2"; return; }
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV3>(env) ) { m_config_vers = "CsPad::ConfigV3"; return; }
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV4>(env) ) { m_config_vers = "CsPad::ConfigV4"; return; }
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV5>(env) ) { m_config_vers = "CsPad::ConfigV5"; return; }

  MsgLog(name(), warning, "CsPad::ConfigV2 - V5 is not available in this run.");
}

//--------------------

void 
CSPadConfigPars::setCSPadConfigParsFromEvent(PSEvt::Event& evt)
{
  if ( getCSPadConfigFromDataForType <Psana::CsPad::DataV1, Psana::CsPad::ElementV1> (evt) ) { m_config_vers = "CsPad::ConfigV1"; return; }
  if ( getCSPadConfigFromDataForType <Psana::CsPad::DataV2, Psana::CsPad::ElementV2> (evt) ) { m_config_vers = "CsPad::ConfigV1"; return; }

  MsgLog(name(), warning, "getCSPadConfigFromData(...): Psana::CsPad::DataV# / ElementV# for #=[2-5] is not available in this event.");
}

//--------------------

} // namespace CSPadPixCoords
