#ifndef PSENV_ENV_H
#define PSENV_ENV_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Env.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <boost/scoped_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEnv/ConfigStore.h"
#include "PSEnv/EpicsStore.h"
#include "RootHistoManager/RootHMgr.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSEnv {

/**
 *  @brief Class representing an environment object for psana jobs.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class Env : boost::noncopyable {
public:

  // Default constructor
  Env (const std::string& jobName) ;

  // Destructor
  ~Env () ;

  /// Return job name
  const std::string& jobName() const { return m_jobName; }
  
  /// Access Configuration Store
  ConfigStore& configStore() { return *m_cfgStore; }

  /// Access EPICS Store
  EpicsStore& epicsStore() { return *m_epicsStore; }

  /// Access ROT histogram manager
  RootHistoManager::RootHMgr& rhmgr() { return *m_rhmgr; }

protected:

private:

  // Data members
  std::string m_jobName;
  boost::scoped_ptr<ConfigStore> m_cfgStore;
  boost::scoped_ptr<EpicsStore> m_epicsStore;
  boost::scoped_ptr<RootHistoManager::RootHMgr> m_rhmgr;
  
};

} // namespace PSEnv

#endif // PSENV_ENV_H
