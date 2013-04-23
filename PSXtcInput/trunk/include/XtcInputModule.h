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
#include <string>
#include <vector>
#include <boost/thread/thread.hpp>
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

class XtcInputModule : public XtcInputModuleBase {
public:

  /// Constructor takes the name of the module.
  XtcInputModule (const std::string& name) ;

  // Destructor
  virtual ~XtcInputModule () ;

protected:
  
private:

  // Initialization method for external datagram source
  virtual void initDgramSource();

  // Get the next datagram from some external source
  virtual XtcInput::Dgram nextDgram();

  // Data members
  boost::scoped_ptr<XtcInput::DgramQueue> m_dgQueue;  ///< Input datagram queue
  boost::scoped_ptr<boost::thread> m_readerThread;    ///< Thread which does datagram reading
  std::vector<std::string> m_fileNames;               ///< List of file names/datasets to read data from
};

} // namespace PSXtcInput

#endif // PSXTCINPUT_XTCINPUTMODULE_H
