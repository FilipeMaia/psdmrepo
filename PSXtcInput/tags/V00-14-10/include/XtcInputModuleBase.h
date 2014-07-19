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
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "psana/InputModule.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSXtcInput/DamagePolicy.h"
#include "PSXtcInput/IDatagramSource.h"
#include "psddl_pds2psana/XtcConverter.h"
#include "pdsdata/xtc/TransitionId.hh"
#include "pdsdata/xtc/ClockTime.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

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

  /**
   *  @brief Constructor takes the name of the module and instance of datagram source.
   *
   *  If noSkip parameter is set to true then psana parameters "skip-events", "events",
   *  "skip-epics" and "l3t-accept-only" are ignored. This is intended for use with
   *  slave worker process in multi-process mode where all skipping is done on master
   *  side.
   *
   */
  XtcInputModuleBase (const std::string& name,
      const boost::shared_ptr<IDatagramSource>& dgsource,
      bool noSkip = false);

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
  
protected:

  /// Protected since the Indexing input module needs access to it.
  boost::shared_ptr<IDatagramSource> m_dgsource;      ///< Datagram source instance

private:

  DamagePolicy m_damagePolicy;                        ///< Policy instance for damage data
  std::vector<XtcInput::Dgram> m_putBack;             ///< Buffer for put-back datagrams
  psddl_pds2psana::XtcConverter m_cvt;                ///< Data converter object
  Pds::ClockTime m_transitions[Pds::TransitionId::NumberOf];  ///< Timestamps of the observed transitions
  unsigned long m_skipEvents;                         ///< Number of events (L1Accept transitions) to skip
  unsigned long m_maxEvents;                          ///< Number of events (L1Accept transitions) to process
  bool m_skipEpics;                                   ///< If true then skip EPICS-only events
  bool m_l3tAcceptOnly;                               ///< If true then pass only events accepted by L3T
  unsigned long m_l1Count;                            ///< Number of events (L1Accept transitions) seen so far
  long m_eventTagEpicsStore;                          ///< counter to pass to epicsStore
  int m_simulateEOR;                                  ///< if non-zero then simulate endRun/stop
  int m_run;                                          ///< Run number that comes from BeginRun transition (or -1)
};

} // namespace PSXtcInput

#endif // PSXTCINPUT_XTCINPUTMODULEBASE_H
