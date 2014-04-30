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
#include <boost/enable_shared_from_this.hpp>

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

class Env : public boost::enable_shared_from_this<Env>, boost::noncopyable {
public:

  /**
   *  @brief Constructor
   *
   *  @param[in] jobName    Name of the psana job.
   *  @param[in] expNameProvider  Object which provides experiment/instrument names.
   *  @param[in] calibDir  Name of the calibration directory, can include "{exp}"
   *                       and "{instr}" strings which will be replaced with experiment
   *                       and instrument names.
   *  @param[in] aliasMap  Optional instance of the alias map.
   *  @param[in] subproc   Subprocess number.
   */
  Env (const std::string& jobName,
      const boost::shared_ptr<IExpNameProvider>& expNameProvider,
      const std::string& calibDir,
      const boost::shared_ptr<PSEvt::AliasMap>& aliasMap,
      int subproc) ;

  // Destructor
  ~Env () ;

  /** 
   *  @brief Returns name of the framework. 
   *  
   *  This method is supposed to be defined across different frameworks. 
   *  It returns the name of the current framework, e.g. when client code runs 
   *  inside pyana framework it will return string "pyana", inside  psana framework 
   *  it will return "psana". This method should be used as a primary mechanism for 
   *  distinguishing between different frameworks in cases when client needs to 
   *  execute framework-specific code. This method is not very useful in C++ as
   *  we have only one C++ framework, but is more useful in Python code which 
   *  needs to run inside both pyana and psana.
   */
  const std::string& fwkName() const { return m_fwkName; }
  
  /// Returns job name.
  const std::string& jobName() const { return m_jobName; }
  
  /// Returns combination of job name and subprocess index as a string
  /// which is unique for all subprocesses in a job..
  const std::string& jobNameSub() const { return m_jobNameSub; }

  /// Returns instrument name
  const std::string& instrument() const { return m_expNameProvider->instrument(); }

  /// Returns experiment name
  const std::string& experiment() const { return m_expNameProvider->experiment(); }

  /// Returns experiment number or 0
  const unsigned expNum() const { return m_expNameProvider->expNum(); }

  /// Returns sub-process number. In case of multi-processing job it will be a non-negative number
  /// ranging from 0 to a total number of sub-processes. In case of single-process job it will return -1.
  const int subprocess() const { return m_subproc; }

  /// Returns that name of the calibration directory for current
  /// instrument/experiment.
  const std::string& calibDir() const;

  /// Access to Configuration Store object.
  EnvObjectStore& configStore() { return *m_cfgStore; }

  /// Access to Calibration Store object.
  EnvObjectStore& calibStore() { return *m_calibStore; }

  /// Access to EPICS Store object.
  EpicsStore& epicsStore() { return *m_epicsStore; }

  /// Access to alias map.
  boost::shared_ptr<PSEvt::AliasMap> aliasMap() { return m_aliasMap; }

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
  std::string m_fwkName;   ///< Framework name
  std::string m_jobName;   ///< Job name
  std::string m_jobNameSub;   ///< Job name with sub-process index
  boost::shared_ptr<PSEvt::AliasMap> m_aliasMap;  ///< Alias map instance
  boost::shared_ptr<EnvObjectStore> m_cfgStore;   ///< Pointer to Configuration Store
  boost::shared_ptr<EnvObjectStore> m_calibStore;   ///< Pointer to Calibration Store
  boost::shared_ptr<EpicsStore> m_epicsStore;  ///< Pointer to EPICS Store
  boost::scoped_ptr<RootHistoManager::RootHMgr> m_rhmgr;  ///< Pointer to ROOT histogram manager
  boost::scoped_ptr<PSHist::HManager> m_hmgr;  ///< Pointer to ROOT histogram manager
  boost::shared_ptr<IExpNameProvider> m_expNameProvider; ///< Object which provides experiment and instrument names
  mutable std::string m_calibDir;              ///< Name of the calibration directory
  mutable bool m_calibDirSetup;                ///< Flag set to true after calibration directory name is fixed
  int m_subproc;
  
};

} // namespace PSEnv

#endif // PSENV_ENV_H
