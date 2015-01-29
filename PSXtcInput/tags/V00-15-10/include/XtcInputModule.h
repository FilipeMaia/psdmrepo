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

};

} // namespace PSXtcInput

#endif // PSXTCINPUT_XTCINPUTMODULE_H
