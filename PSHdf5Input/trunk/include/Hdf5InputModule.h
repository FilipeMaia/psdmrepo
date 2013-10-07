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
#include <vector>
#include <string>
#include <boost/scoped_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "psana/InputModule.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psddl_hdf2psana/HdfConverter.h"
#include "PSEvt/EventId.h"

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
 *  @brief Package with the implementation of psana input module for HDF5 files.
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
  virtual void beginJob(Event& evt, Env& env);

  /// Method which is called with event data
  virtual Status event(Event& evt, Env& env);

  /// Method which is called once at the end of the job
  virtual void endJob(Event& evt, Env& env);

protected:

  // Store EPICS data in environment
  void fillEpics(const Hdf5IterData& data, Env& env);

  // Store event ID object
  void fillEventId(const Hdf5IterData& data, Event& evt);

  // Store data objects in even and environment
  void fillEventEnv(const Hdf5IterData& data, Event& evt, Env& env);

private:

  std::vector<std::string> m_datasets;                ///< List of file names/datasets to read data from
  boost::scoped_ptr<Hdf5FileListIter> m_iter;
  psddl_hdf2psana::HdfConverter m_cvt;
  unsigned long m_skipEvents;                         ///< Number of events to skip
  unsigned long m_maxEvents;                          ///< Number of events to process
  bool m_l3tAcceptOnly;                               ///< If true then pass only events accepted by L3T
  unsigned long m_l1Count;                            ///< Number of events seen so far
  int m_simulateEOR;                                  ///< if non-zero then simulate endRun/stop
  boost::shared_ptr<PSEvt::EventId> m_evtId;          ///< remembered EventId for simulated EOR

};

} // namespace PSHdf5Input

#endif // PSHDF5INPUT_HDF5INPUTMODULE_H
