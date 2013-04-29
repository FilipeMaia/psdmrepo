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
 *  shared memory. Psana uses this module instead of regular XTC (or HDF5)
 *  input module when it encounters @e shmem keyword in the input dataset:
 *
 *  @code
 *  % psana -c config.cfg shmem=name.0
 *  @endcode
 *
 *  Required value for @e shmem keyword consists of the shared memory tag
 *  name and a client index (number) separated by dot, the meaning of these
 *  parameters is defined by the DAQ shared memory server and should be
 *  known to people who setup the infrastructure.
 *
 *  In addition to @e shmem keyword dataset specification can optionally
 *  contain @e stop keyword with a value that specifies stop condition.
 *  Possible values for this keyword are:
 *    - @e unmap - psana stops when @c UnMap transition occurs
 *    - @e unconfigure - psana stops when @c UnConfigure transition occurs
 *    - @e endrun - psana stops when @c EndRun transition occurs
 *    - @e endcalibcycle - psana stops when @c EndCalibCycle transition occurs
 *    - @e none - psana runs forever until killed
 *    - @e no - same as @e none
 *
 *  By default if @e stop keyword is not specified then it is equivalent to
 *  @e stop=endrun and psana will stop at the end of run.
 *
 *  Example of starting non-stopping job:
 *
 *  @code
 *  % psana -c config.cfg shmem=name.0:stop=none
 *  @endcode
 *
 *  @note This software was developed for the LCLS project.  If you use all or
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
