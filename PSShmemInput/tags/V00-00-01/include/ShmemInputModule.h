#ifndef PSSHMEMINPUT_SHMEMINPUTMODULE_H
#define PSSHMEMINPUT_SHMEMINPUTMODULE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ShmemInputModule.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <boost/thread.hpp>
#include <boost/scoped_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "PSXtcInput/XtcInputModuleBase.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/Dgram.h"

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
 *  @brief Psana input module for shared memory.
 *
 *  This class implements psana input module which reads XTC data from
 *  shared memory.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class ShmemInputModule : public PSXtcInput::XtcInputModuleBase {
public:

  /// Constructor takes the name of the module.
  ShmemInputModule (const std::string& name) ;

  // Destructor
  virtual ~ShmemInputModule();

protected:

private:

  // Initialization method for external datagram source
  virtual void initDgramSource();

  // Get the next datagram from some external source
  virtual XtcInput::Dgram nextDgram();

  boost::scoped_ptr<XtcInput::DgramQueue> m_dgQueue;  ///< Input datagram queue
  boost::scoped_ptr<boost::thread> m_readerThread;    ///< Thread which does datagram reading

};

} // namespace PSShmemInput

#endif // PSSHMEMINPUT_SHMEMINPUTMODULE_H
