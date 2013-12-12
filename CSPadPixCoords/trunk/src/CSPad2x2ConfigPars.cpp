//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPad2x2ConfigPars...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "CSPadPixCoords/CSPad2x2ConfigPars.h"

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

CSPad2x2ConfigPars::CSPad2x2ConfigPars ()
{
  m_source = PSEvt::Source("DetInfo(:Cspad2x2)");
  setCSPad2x2ConfigParsDefault();
}

CSPad2x2ConfigPars::CSPad2x2ConfigPars (PSEvt::Source source)
    : m_source(source)
{
  setCSPad2x2ConfigParsDefault();
}

CSPad2x2ConfigPars::CSPad2x2ConfigPars ( uint32_t roiMask )
{
    setCSPad2x2ConfigParsDefault();
    m_roiMask        = roiMask;
    m_num2x1Stored   = getNum2x1InMask(m_roiMask);
    m_numAsicsStored = N2x1 * m_num2x1Stored;
}

//--------------
// Destructor --
//--------------

CSPad2x2ConfigPars::~CSPad2x2ConfigPars ()
{
}

//--------------------

uint32_t 
CSPad2x2ConfigPars::getNum2x1InMask(uint32_t mask)
{
  uint32_t num2x1InMask=0;
  for(uint32_t sect=0; sect < 8; sect++) {
    if( mask & (1<<sect) ) num2x1InMask++; 
  }
  return num2x1InMask;
}

//--------------------

void 
CSPad2x2ConfigPars::setCSPad2x2ConfigParsDefault()
{
    m_roiMask        = 03; // or 3;
    m_num2x1Stored   = getNum2x1InMask(m_roiMask);
    m_numAsicsStored = 2 * N2x1;      
    m_config_vers    = "N/A yet";
    m_data_vers      = "N/A yet";
    std::fill_n(&m_common_mode[0], int(N2x1), float(0));
    m_is_set_for_evt = false;
    m_is_set_for_env = false;
    m_is_set         = false;
}

//--------------------

/// Print configuration parameters
void 
CSPad2x2ConfigPars::printCSPad2x2ConfigPars()
{
  WithMsgLog(name(), info, log) {
    log << "Config pars from "  << m_config_vers
        << " and "              << m_data_vers
        << "\nN configs found:" << m_count_cfg
        << "  roiMask="         << m_roiMask
        << "  num2x1Stored="    << m_num2x1Stored
        << "  numAsicsStored="  << m_numAsicsStored
        << "  common_mode="     << m_common_mode[0] << ", " << m_common_mode[1]
        << "  is_set_for_evt:"  << m_is_set_for_evt    
        << "  is_set_for_env:"  << m_is_set_for_env    
        << "  is_set:"          << m_is_set    
        << "\n";
  }  
}

//--------------------

bool
CSPad2x2ConfigPars::setCSPad2x2ConfigPars(PSEvt::Event& evt, PSEnv::Env& env)
{
  if ( ! m_is_set_for_env ) { m_is_set_for_env = setCSPad2x2ConfigParsFromEnv(env); }
  if ( ! m_is_set_for_evt ) { m_is_set_for_evt = setCSPad2x2ConfigParsFromEvent(evt); }
  m_is_set = m_is_set_for_env && m_is_set_for_evt;
  return m_is_set;
}

//--------------------

bool 
CSPad2x2ConfigPars::setCSPad2x2ConfigParsFromEnv(PSEnv::Env& env)
{
  m_count_cfg = 0; 
  if ( getConfigParsForType <Psana::CsPad2x2::ConfigV1> (env) ) { m_config_vers = "CsPad2x2::ConfigV1"; return true; }
  if ( getConfigParsForType <Psana::CsPad2x2::ConfigV2> (env) ) { m_config_vers = "CsPad2x2::ConfigV2"; return true; }

  MsgLog(name(), warning, "CsPad2x2::ConfigV1 - V2 is not available in this event...");
  //terminate();
  return false;
}

//--------------------

bool 
CSPad2x2ConfigPars::setCSPad2x2ConfigParsFromEvent(PSEvt::Event& evt)
{
  if ( getCSPadConfigFromDataForType <Psana::CsPad2x2::ElementV1> (evt) ) { m_data_vers = "CsPad2x2::ElementV1"; return true; }

  MsgLog(name(), warning, "setCSPad2x2ConfigParsFromEvent(...): Psana::CsPad2x2::ElementV1 is not available in this event...");
  return false;
}

//--------------------

} // namespace CSPadPixCoords
