#ifndef PSXTCMPINPUT_XTCMPMASTERINPUTBASE_H
#define PSXTCMPINPUT_XTCMPMASTERINPUTBASE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcMPMasterInputBase.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "psana/InputModule.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSXtcInput/IDatagramSource.h"
#include "pdsdata/xtc/TransitionId.hh"
#include "pdsdata/xtc/ClockTime.hh"
#include "psana/MPWorkerId.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSXtcMPInput {

/**
 *  @ingroup PSXtcMPInput
 *  
 *  @brief Psana input module for reading XTC files, used by master process in
 *  multi-process mode.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class XtcMPMasterInputBase : public psana::InputModule {
public:

  /// Constructor takes the name of the module and instance of datagram source
  XtcMPMasterInputBase (const std::string& name, const boost::shared_ptr<PSXtcInput::IDatagramSource>& dgsource);

  // Destructor
  virtual ~XtcMPMasterInputBase () ;

  /// Method which is called once at the beginning of the job
  virtual void beginJob(Event& evt, Env& env);

  /// Method which is called with event data
  virtual Status event(Event& evt, Env& env);

  /// Method which is called once at the end of the job
  virtual void endJob(Event& evt, Env& env);

protected:
  
  /// send datagram to workers
  void send(const XtcInput::Dgram& dg);

  /// send datagram to one worker
  void sendToWorker(int workerId, const XtcInput::Dgram& dg);

  /// if datagram has EPICs data then remember it in case it may be needed later
  void memorizeEpics(const XtcInput::Dgram& dg);

private:

  boost::shared_ptr<PSXtcInput::IDatagramSource> m_dgsource;      ///< Datagram source instance
  XtcInput::Dgram m_putBack;                          ///< Buffer for one put-back datagram
  Pds::ClockTime m_transitions[Pds::TransitionId::NumberOf];  ///< Timestamps of the observed transitions
  unsigned long m_skipEvents;                         ///< Number of events (L1Accept transitions) to skip
  unsigned long m_maxEvents;                          ///< Number of events (L1Accept transitions) to process
  bool m_skipEpics;                                   ///< If true then skip EPICS-only events
  bool m_l3tAcceptOnly;                               ///< If true then pass only events accepted by L3T
  int m_fdReadyPipe;                                  ///< FD for master's end of the ready pipe
  unsigned long m_l1Count;                            ///< Number of events (L1Accept transitions) seen so far
  std::map<Pds::Src, XtcInput::Dgram> m_epicsDgs;     ///< datagrams that contain Epics stuff from previous events
  std::map<int, psana::MPWorkerId> m_workers;         ///< workers indexed by worker id
};

} // namespace PSXtcMPInput

#endif // PSXTCMPINPUT_XTCMPMASTERINPUTBASE_H
