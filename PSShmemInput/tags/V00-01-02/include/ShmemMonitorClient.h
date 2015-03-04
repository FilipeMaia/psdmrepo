#ifndef PSSHMEMINPUT_SHMEMMONITORCLIENT_H
#define PSSHMEMINPUT_SHMEMMONITORCLIENT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ShmemMonitorClient.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------
#include "pdsdata/app/XtcMonitorClient.hh"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/TransitionId.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace XtcInput {
class DgramQueue;
}

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSShmemInput {

/// @addtogroup PSShmemInput

/**
 *  @ingroup PSShmemInput
 *
 *  @brief Implementation of XtcMonitorClient which pushes data into a queue.
 *
 *  Instance of this class is supposed to run in a separate thread so its
 *  interface is suitable for use with boost::thread class, for example.
 *  It is also possible to use this class without starting new thread by
 *  calling @c operator()() like in this example:
 *
 *  @code
 *    ShmemMonitorClient client(...);
 *    // This will run until stop transition is met
 *    client();
 *  @endcode
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class ShmemMonitorClient : public Pds::XtcMonitorClient {
public:

  // Default constructor
  ShmemMonitorClient(const std::string& tag, int index, XtcInput::DgramQueue& queue,
      Pds::TransitionId::Value stopTr);

  // this is the "run" method used by the Boost.thread
  void operator() () ;

  // overriding base class method
  virtual int processDgram(Pds::Dgram* dg);

protected:

private:

  std::string m_tag;                 ///< Shared memory tag
  int m_index;                       ///< Client index
  XtcInput::DgramQueue& m_queue;     ///< Ouput queue for datagrams
  Pds::TransitionId::Value m_stopTr; ///< Transition which should stop event loop

};

} // namespace PSShmemInput

#endif // PSSHMEMINPUT_SHMEMMONITORCLIENT_H
