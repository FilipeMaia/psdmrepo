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
#include "PSEnv/EnvObjectStore.h"
#include "PSEnv/EpicsStore.h"
#include "PSEnv/IExpNameProvider.h"
#include "RootHistoManager/RootHMgr.h"
#include "PSHist/HManager.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------


/**
 *  @defgroup PSEnv  PSEnv package
 *  
 *  @brief PSEnv package contains classes which provide storage and 
 *  access to non-event data in the context of psana framework.
 *  
 *  The core of the package is the Env class which stores other
 *  types of objects such as configuration data, EPICS data, etc. 
 */

namespace PSEnv {

/**
 *  @ingroup PSEnv
 *  
 *  @brief Class representing an environment object for psana jobs.
 *  
 *  Environment object stores non-event data such as configuration objects, 
 *  EPICS data, histogram manager, plus some job-specific information
 *  which does not change from event to event.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Andrei Salnikov
 */

class Env : boost::noncopyable {
public:

  /**
   *  @brief Constructor
   *
   *  @param[in] jobName    Name of the psana job.
   *  @param[in] expNameProvider  Object which provides experiment/instrument names.
   *  @param[in] calibDir  Name of the calibration directory, can include "{exp}"
   *                       and "{instr}" strings which will be replaced with experiment
   *                       and instrument names.
   */
  Env (const std::string& jobName,
      const boost::shared_ptr<IExpNameProvider>& expNameProvider,
      const std::string& calibDir) ;

  // Destructor
  ~Env () ;

  /// Returns job name.
  const std::string& jobName() const { return m_jobName; }
  
  /// Returns instrument name
  const std::string& instrument() const { return m_expNameProvider->instrument(); }

  /// Returns experiment name
  const std::string& experiment() const { return m_expNameProvider->experiment(); }

  /// Returns that name of the calibration directory for current
  /// instrument/experiment.
  const std::string& calibDir() const;

  /// Access to Configuration Store object.
  EnvObjectStore& configStore() { return *m_cfgStore; }

  /// Access to Calibration Store object.
  EnvObjectStore& calibStore() { return *m_calibStore; }

  /// Access to EPICS Store object.
  EpicsStore& epicsStore() { return *m_epicsStore; }

  /**
   *  @brief DEPRECATED: Access to ROOT histogram manager.
   *  
   *  @deprecated Use hmgr() instead.
   */
  RootHistoManager::RootHMgr& rhmgr() { return *m_rhmgr; }

  /// Access to histogram manager.
  PSHist::HManager& hmgr();

protected:

private:

  // Data members
  std::string m_jobName;   ///< Job name
  boost::scoped_ptr<EnvObjectStore> m_cfgStore;   ///< Pointer to Configuration Store
  boost::scoped_ptr<EnvObjectStore> m_calibStore;   ///< Pointer to Calibration Store
  boost::scoped_ptr<EpicsStore> m_epicsStore;  ///< Pointer to EPICS Store
  boost::scoped_ptr<RootHistoManager::RootHMgr> m_rhmgr;  ///< Pointer to ROOT histogram manager
  boost::scoped_ptr<PSHist::HManager> m_hmgr;  ///< Pointer to ROOT histogram manager
  boost::shared_ptr<IExpNameProvider> m_expNameProvider; ///< Object which provides experiment and instrument names
  mutable std::string m_calibDir;              ///< Name of the calibration directory
  mutable bool m_calibDirSetup;                ///< Flag set to true after calibration directory name is fixed
  
};

} // namespace PSEnv

#endif // PSENV_ENV_H
