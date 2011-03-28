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
#include <boost/scoped_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEnv/ConfigStore.h"
#include "PSEnv/EpicsStore.h"

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
  Env () ;

  // Destructor
  ~Env () ;
  
  /// Access Configuration Store
  ConfigStore& configStore() { return *m_cfgStore; }

  /// Access EPICS Store
  EpicsStore& epicsStore() { return *m_epicsStore; }

protected:

private:

  // Data members
  boost::scoped_ptr<ConfigStore> m_cfgStore;
  boost::scoped_ptr<EpicsStore> m_epicsStore;
  
};

} // namespace PSEnv

#endif // PSENV_ENV_H
