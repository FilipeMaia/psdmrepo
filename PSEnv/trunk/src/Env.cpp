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
#include <dlfcn.h>
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEvt/ProxyDict.h"
#include "PSEvt/ProxyDictHist.h"
#include "MsgLogger/MsgLogger.h"
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
    const std::string& calibDir,
    const boost::shared_ptr<PSEvt::AliasMap>& aliasMap,
    int subproc)
  : m_fwkName("psana")
  , m_jobName(jobName)
  , m_jobNameSub(jobName)
  , m_aliasMap(aliasMap)
  , m_cfgStore()
  , m_calibStore()
  , m_epicsStore(boost::make_shared<EpicsStore>())
  , m_hmgr()
  , m_firstHmgrCall(true)
  , m_expNameProvider(expNameProvider)
  , m_calibDir(calibDir)
  , m_calibDirSetup(false)
  , m_subproc(subproc)
{
  // instantiate dictionary for config store and store itself
  boost::shared_ptr<PSEvt::ProxyDictHist> cfgDict(new PSEvt::ProxyDictHist(m_aliasMap));
  m_cfgStore = boost::make_shared<EnvObjectStore>(cfgDict);
  
  // instantiate dictionary for calib store and store itself
  boost::shared_ptr<PSEvt::ProxyDictHist> calibDict(new PSEvt::ProxyDictHist(m_aliasMap));
  m_calibStore = boost::make_shared<EnvObjectStore>(calibDict);
  
  if (subproc >= 0) {
    m_jobNameSub += boost::lexical_cast<std::string>(subproc);
  }
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
boost::shared_ptr<PSHist::HManager> 
Env::hmgr()
{
  typedef PSHist::HManager *(*FnPSHistFromConstCharPtr)(const char *);
  if (not m_hmgr.get() and m_firstHmgrCall) {
    m_firstHmgrCall = false;
    const char * histPackage = "libRootHist.so";  // this could become a psana parameter
    const char * histFactoryFunction = "CREATE_PSHIST_HMANAGER_FROM_CONST_CHAR_PTR"; // this fixed,
                                // but we could look for factory functions taking other parameters as well
    void* histLibHandle = dlopen(histPackage, RTLD_NOW | RTLD_GLOBAL);
    if (histLibHandle) {
      FnPSHistFromConstCharPtr histFactorySymbol = (FnPSHistFromConstCharPtr)dlsym(histLibHandle,
                                                                                   histFactoryFunction);
      if (histFactorySymbol) {
        // Instantiate manager
        std::string rfname = m_jobName + ".root";
        boost::shared_ptr<PSHist::HManager> hmgr(histFactorySymbol(rfname.c_str()));
        if (hmgr) {
          m_hmgr = hmgr;
        } else {
          MsgLogRoot(error, "Env::hmgr() factory function to create PSHist::HManager in package "
                     << histPackage << " found and called, but it did not return a valid object.");
        }
      } else {
        MsgLogRoot(error, "Env::hmgr() factory function=" << histFactoryFunction
                   << " to create PSHist::Hmanager in package "
                   << histPackage 
                   << " not found.");
      } 
    } else {
      MsgLogRoot(warning, "Env::hmgr() package to create PSHist::Hmanager = " 
                 << histPackage 
                 << " not found.");
    }
  }
  return m_hmgr;
}


} // namespace PSEnv
