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
  
  /**
   *  @brief Fill event from list of datagrams.
   *
   * Datagrams should be sorted with DAQ streams first, Control streams last. If transition is 
   * Configure or BeginCalibCycle, uses first DAQ stream (skips others) and all control streams. 
   * For all other transitions all Dgrams are processed.
   */
  void fillEvent(const std::vector<XtcInput::Dgram>& dgList, Event& evt, Env& env);

  /// Fill event with datagram contents
  void fillEvent(const XtcInput::Dgram& dg, Event& evt, Env& env);
  
  /// Fill event with EventId information
  void fillEventId(const XtcInput::Dgram& dg, Event& evt);

  /// Fill event with Datagram list
  void fillEventDgList(const std::vector<XtcInput::Dgram>& dgList, Event& evt);

  /**
   * @brief Fill env from list of datagrams. 
   *
   * Datagrams should be sorted with DAQ streams first, Control streams last. If transition is 
   * Configure or BeginCalibCycle, uses first DAQ stream (skips others) and all control streams. 
   * For all other transitions all Dgrams are processed.
   */
  void fillEnv(const std::vector<XtcInput::Dgram> &dgList, Env &env);

  /// Fill environment with datagram contents
  void fillEnv(const XtcInput::Dgram& dg, Env& env);

  /// Protected since the Indexing input module needs access to it.
  boost::shared_ptr<IDatagramSource> m_dgsource;      ///< Datagram source instance

  /// Protected to allow specific input modules to ignore settings they
  /// don't support
  void skipEvents(unsigned long nskip) {m_skipEvents = nskip;}
  void maxEvents(unsigned long nmax)   {m_maxEvents = nmax;}
  void skipEpics(bool skip)            {m_skipEpics = skip;}
  void l3tAcceptOnly(bool l3tAccept)   {m_l3tAcceptOnly = l3tAccept;}
  unsigned long skipEvents() const     {return m_skipEvents;}
  unsigned long maxEvents()  const     {return m_maxEvents;}
  bool skipEpics()           const     {return m_skipEpics;}
  bool l3tAcceptOnly()       const     {return m_l3tAcceptOnly;}

private:

  DamagePolicy m_damagePolicy;                        ///< Policy instance for damage data
  std::vector<XtcInput::Dgram> m_putBack;             ///< Buffer for put-back datagrams
  psddl_pds2psana::XtcConverter m_cvt;                ///< Data converter object
  Pds::ClockTime m_transitions[Pds::TransitionId::NumberOf];  ///< Timestamps of the observed transitions
  unsigned long m_skipEvents;                         ///< Number of events (L1Accept transitions) to skip
  unsigned long m_maxEvents;                          ///< Number of events (L1Accept transitions) to process
  bool m_skipEpics;                                   ///< If true then skip EPICS-only events
  bool m_l3tAcceptOnly;                               ///< If true then pass only events accepted by L3T
  int m_firstControlStream;                           ///< Starting index of control streams
  unsigned long m_l1Count;                            ///< Number of events (L1Accept transitions) seen so far
  long m_eventTagEpicsStore;                          ///< counter to pass to epicsStore
  int m_simulateEOR;                                  ///< if non-zero then simulate endRun/stop
  int m_run;                                          ///< Run number that comes from BeginRun transition (or -1)
  bool m_liveMode;                                    ///< true if live mode specified in psana files option
  unsigned m_liveTimeOut;                             ///< live timeout value from config
};

} // namespace PSXtcInput

#endif // PSXTCINPUT_XTCINPUTMODULEBASE_H
