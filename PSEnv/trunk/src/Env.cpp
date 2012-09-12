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
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEvt/ProxyDict.h"
#include "RootHist/RootHManager.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  void gsub(std::string& str, const std::string& substr, const std::string& replacement)
  {
    for (std::string::size_type p = str.find(substr); p != std::string::npos; p = str.find(substr)) {
      str.replace(p, substr.size(), replacement);
    }
  }


}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSEnv {

//----------------
// Constructors --
//----------------
Env::Env (const std::string& jobName,
    const boost::shared_ptr<IExpNameProvider>& expNameProvider,
    const std::string& calibDir)
  : m_jobName(jobName)
  , m_cfgStore()
  , m_calibStore()
  , m_epicsStore(boost::make_shared<EpicsStore>())
  , m_rhmgr()
  , m_hmgr()
  , m_expNameProvider(expNameProvider)
  , m_calibDir(calibDir)
  , m_calibDirSetup(false)
{
  // instantiate dictionary for config store and store itself
  boost::shared_ptr<PSEvt::ProxyDict> cfgDict(new PSEvt::ProxyDict());
  m_cfgStore = boost::make_shared<EnvObjectStore>(cfgDict);

  // instantiate dictionary for calib store and store itself
  boost::shared_ptr<PSEvt::ProxyDict> calibDict(new PSEvt::ProxyDict());
  m_calibStore = boost::make_shared<EnvObjectStore>(calibDict);
  
  // make root file name
  std::string rfname = jobName + "-rhmgr.root";
  m_rhmgr.reset(new RootHistoManager::RootHMgr(rfname));
}

//--------------
// Destructor --
//--------------
Env::~Env ()
{
  // save histograms
  if (m_hmgr.get()) m_hmgr->write();
}

// Returns that name of the calibration directory for current
// instrument/experiment.
const std::string&
Env::calibDir() const
{
  if (not m_calibDirSetup) {
    ::gsub(m_calibDir, "{exp}", experiment());
    ::gsub(m_calibDir, "{instr}", instrument());
    m_calibDirSetup = true;
  }

  return m_calibDir;
}

// Access to histogram manager.
PSHist::HManager& 
Env::hmgr()
{
  if (not m_hmgr.get()) {
    // Instantiate manager
    std::string rfname = m_jobName + ".root";
    m_hmgr.reset(new RootHist::RootHManager(rfname));
  }
  return *m_hmgr;
}


} // namespace PSEnv
