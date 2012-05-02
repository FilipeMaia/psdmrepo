#ifndef PSHDF5INPUT_HDF5INPUTMODULE_H
#define PSHDF5INPUT_HDF5INPUTMODULE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Hdf5InputModule.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/scoped_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "psana/InputModule.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psddl_hdf2psana/HdfConverter.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace PSHdf5Input {
class Hdf5FileListIter;
class Hdf5IterData;
}

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  @defgroup PSHdf5Input PSHdf5Input package
 *  
 *  @brief Package with the implementation if psana input module for HDF5 files.
 *  
 */

namespace PSHdf5Input {

/**
 *  @ingroup PSHdf5Input
 *  
 *  @brief Psana input module for reading HDF5 files.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Hdf5InputModule : public psana::InputModule {
public:

    /// Constructor takes the name of the module.
  Hdf5InputModule (const std::string& name) ;

  // Destructor
  virtual ~Hdf5InputModule () ;

  /// Method which is called once at the beginning of the job
  virtual void beginJob(Env& env);

  /// Method which is called with event data
  virtual Status event(Event& evt, Env& env);

  /// Method which is called once at the end of the job
  virtual void endJob(Env& env);

protected:

  // Store config object in environment
  void fillConfig(const Hdf5IterData& data, Env& env);

  // Store EPICS data in environment
  void fillEpics(const Hdf5IterData& data, Env& env);

  // Store event ID object
  void fillEventId(const Hdf5IterData& data, Event& evt);

  // Store event data objects
  void fillEvent(const Hdf5IterData& data, Event& evt, Env& env);

private:

  boost::scoped_ptr<Hdf5FileListIter> m_iter;
  psddl_hdf2psana::HdfConverter m_cvt;
  unsigned long m_skipEvents;                         ///< Number of events to skip
  unsigned long m_maxEvents;                          ///< Number of events to process
  unsigned long m_l1Count;                            ///< Number of events seen so far

};

} // namespace PSHdf5Input

#endif // PSHDF5INPUT_HDF5INPUTMODULE_H
