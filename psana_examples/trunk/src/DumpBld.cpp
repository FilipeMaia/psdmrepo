//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpBld...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpBld.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "pdsdata/xtc/BldInfo.hh"
#include "psddl_psana/bld.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpBld)

namespace {
  
  // name of the logger to be used with MsgLogger
  const char* logger = "DumpBld"; 
  
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpBld::DumpBld (const std::string& name)
  : Module(name)
{
  m_ebeamSrc = configStr("eBeamSource", "BldInfo(EBeam)");
  m_cavSrc = configStr("phaseCavSource", "BldInfo(PhaseCavity)");
  m_feeSrc = configStr("feeSource", "BldInfo(FEEGasDetEnergy)");
}

//--------------
// Destructor --
//--------------
DumpBld::~DumpBld ()
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpBld::event(Event& evt, Env& env)
{
  shared_ptr<Psana::Bld::BldDataEBeam> ebeam = evt.get(m_ebeamSrc);
  if (not ebeam.get()) {
    MsgLog(logger, info, name() << ": Bld::BldDataEBeam not found");    
  } else {
    MsgLog(logger, info, name() << ": Bld::BldDataEBeam: ebeamCharge=" << ebeam->ebeamCharge()
         << " ebeamL3Energy=" << ebeam->ebeamL3Energy());
  }

  shared_ptr<Psana::Bld::BldDataPhaseCavity> cav = evt.get(m_cavSrc);
  if (not cav.get()) {
    MsgLog(logger, info, name() << ": Bld::BldDataPhaseCavity not found");    
  } else {
    MsgLog(logger, info, name() << ": Bld::BldDataPhaseCavity: fitTime1=" << cav->fitTime1()
         << " fitTime2=" << cav->fitTime2()
         << " charge1=" << cav->charge1()
         << " charge2=" << cav->charge2()
         );
  }
  
  shared_ptr<Psana::Bld::BldDataFEEGasDetEnergy> fee = evt.get(m_feeSrc);
  if (not fee.get()) {
    MsgLog(logger, info, name() << ": Bld::BldDataFEEGasDetEnergy not found");    
  } else {
    MsgLog(logger, info, name() << ": Bld::BldDataFEEGasDetEnergy: f_11_ENRC=" << fee->f_11_ENRC()
         << " f_12_ENRC=" << fee->f_12_ENRC()
         << " f_21_ENRC=" << fee->f_21_ENRC()
         << " f_22_ENRC=" << fee->f_22_ENRC()
         );
  }
}

} // namespace psana_examples
