#ifndef PSXTCMPINPUT_XTCMPWORKERINPUT_H
#define PSXTCMPINPUT_XTCMPWORKERINPUT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcMPWorkerInput.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "PSXtcInput/XtcInputModuleBase.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSXtcMPInput {

/// @addtogroup PSXtcMPInput

/**
 *  @ingroup PSXtcMPInput
 *
 *  @brief Input module for use in multi-process worker.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class XtcMPWorkerInput : public PSXtcInput::XtcInputModuleBase {
public:

  /// Constructor takes the name of the module.
  XtcMPWorkerInput (const std::string& name) ;

  // Destructor
  virtual ~XtcMPWorkerInput () ;

protected:

private:

};

} // namespace PSXtcMPInput

#endif // PSXTCMPINPUT_XTCMPWORKERINPUT_H
