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
#include "pdsdata/xtc/Dgram.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace XtcInput {
  class DgramQueue;
}


//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSXtcInput {

/**
 *  @brief PSANA module for reading XTC files.
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

class XtcInputModule : public InputModule {
public:

  // Default constructor
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
  
  // Fill event with datagram contents
  void fillEvent(const boost::shared_ptr<Pds::Dgram>& dg, Event& evt);
  
  // Fill environment with datagram contents
  void fillEnv(const boost::shared_ptr<Pds::Dgram>& dg, Env& env);

private:

  // Data members
  boost::scoped_ptr<XtcInput::DgramQueue> m_dgQueue;
  boost::shared_ptr<Pds::Dgram> m_putBack;
  boost::scoped_ptr<boost::thread> m_readerThread;
};

} // namespace PSXtcInput

#endif // PSXTCINPUT_XTCINPUTMODULE_H
