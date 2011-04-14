//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Env...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSEnv/Env.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEvt/ProxyDict.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSEnv {

//----------------
// Constructors --
//----------------
Env::Env (const std::string& jobName)
  : m_jobName(jobName)
  , m_cfgStore()
  , m_epicsStore(new EpicsStore())
  , m_rhmgr()
{
  // instantiate dictionary for config store and store itself
  boost::shared_ptr<PSEvt::ProxyDict> cfgDict(new PSEvt::ProxyDict());
  m_cfgStore.reset(new ConfigStore(cfgDict));
  
  // make root file name
  std::string rfname = jobName + ".root";
  m_rhmgr.reset(new RootHistoManager::RootHMgr(rfname));
}

//--------------
// Destructor --
//--------------
Env::~Env ()
{
}

} // namespace PSEnv
