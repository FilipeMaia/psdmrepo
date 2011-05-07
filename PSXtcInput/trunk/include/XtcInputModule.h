#ifndef PSXTCINPUT_XTCINPUTMODULE_H
#define PSXTCINPUT_XTCINPUTMODULE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcInputModule.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
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

class XtcInputModule : public InputModule {
public:

  /// Constructor takes the name of the module.
  XtcInputModule (const std::string& name) ;

  // Destructor
  virtual ~XtcInputModule () ;

  /// Method which is called once at the beginning of the job
  virtual void beginJob(Env& env);

  /// Method which is called with event data
  virtual Status event(Event& evt, Env& env);

  /// Method which is called once at the end of the job
  virtual void endJob(Env& env);

protected:
  
  /// Fill event with datagram contents
  void fillEvent(const XtcInput::Dgram& dg, Event& evt, Env& env);
  
  /// Fill environment with datagram contents
  void fillEnv(const XtcInput::Dgram& dg, Env& env);

private:

  // Data members
  boost::scoped_ptr<XtcInput::DgramQueue> m_dgQueue;  ///< Input datagram queue
  XtcInput::Dgram m_putBack;                          ///< Buffer for one put-back datagram
  boost::scoped_ptr<boost::thread> m_readerThread;    ///< Thread which does datagram reading
  psddl_pds2psana::XtcConverter m_cvt;                ///< Data converter object
  Pds::ClockTime m_transitions[Pds::TransitionId::NumberOf];  ///< Timestamps of the observed transitions
};

} // namespace PSXtcInput

#endif // PSXTCINPUT_XTCINPUTMODULE_H
