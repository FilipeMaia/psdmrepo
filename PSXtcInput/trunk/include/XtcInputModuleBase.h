#ifndef PSXTCINPUT_XTCINPUTMODULEBASE_H
#define PSXTCINPUT_XTCINPUTMODULEBASE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcInputModuleBase.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <boost/thread/thread.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "psana/InputModule.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/Dgram.h"
#include "psddl_pds2psana/XtcConverter.h"
#include "pdsdata/xtc/TransitionId.hh"
#include "pdsdata/xtc/ClockTime.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace XtcInput {
  class DgramQueue;
}


//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  @defgroup PSXtcInput PSXtcInput package
 *  
 *  @brief Package with the implementation if psana input module for XTC files.
 *  
 */

namespace PSXtcInput {

/**
 *  @ingroup PSXtcInput
 *  
 *  @brief Psana input module for reading XTC files.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class XtcInputModuleBase : public InputModule {
public:

  /// Constructor takes the name of the module.
  XtcInputModuleBase (const std::string& name) ;

  // Destructor
  virtual ~XtcInputModuleBase () ;

  /// Method which is called once at the beginning of the job
  virtual void beginJob(Event& evt, Env& env);

  /// Method which is called with event data
  virtual Status event(Event& evt, Env& env);

  /// Method which is called once at the end of the job
  virtual void endJob(Event& evt, Env& env);

protected:
  
  /// Fill event with datagram contents
  void fillEvent(const XtcInput::Dgram& dg, Event& evt, Env& env);
  
  /// Fill event with EventId information
  void fillEventId(const XtcInput::Dgram& dg, Event& evt);

  /// Fill event with Datagram
  void fillEventDg(const XtcInput::Dgram& dg, Event& evt);

  /// Fill environment with datagram contents
  void fillEnv(const XtcInput::Dgram& dg, Env& env);

private:

  // Initialization method for external datagram source
  virtual void initDgramSource() = 0;

  // Get the next datagram from some external source
  virtual XtcInput::Dgram nextDgram() = 0;

  // Data members
  XtcInput::Dgram m_putBack;                          ///< Buffer for one put-back datagram
  psddl_pds2psana::XtcConverter m_cvt;                ///< Data converter object
  Pds::ClockTime m_transitions[Pds::TransitionId::NumberOf];  ///< Timestamps of the observed transitions
  unsigned long m_skipEvents;                         ///< Number of events (L1Accept transitions) to skip
  unsigned long m_maxEvents;                          ///< Number of events (L1Accept transitions) to process
  bool m_skipEpics;                                   ///< If true then skip EPICS-only events
  unsigned long m_l1Count;                            ///< Number of events (L1Accept transitions) seen so far
  int m_simulateEOR;                                  ///< if non-zero then simulate endRun/stop
};

} // namespace PSXtcInput

#endif // PSXTCINPUT_XTCINPUTMODULEBASE_H
