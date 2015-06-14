#ifndef PSXTCINPUT_XTCINDEXINPUTMODULE_H
#define PSXTCINPUT_XTCINDEXINPUTMODULE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: XtcIndexInputModule.h 7696 2014-02-27 00:40:59Z salnikov@SLAC.STANFORD.EDU $
//
// Description:
//	Class XtcIndexInputModule.
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
#include "PSXtcInput/Index.h"
#include "PSXtcInput/DgramPieces.h"

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
 *  @version $Id: XtcIndexInputModule.h 7696 2014-02-27 00:40:59Z salnikov@SLAC.STANFORD.EDU $
 *
 *  @author Andrei Salnikov
 */

  class XtcIndexInputModule : public XtcInputModuleBase {
public:

  /// Constructor takes the name of the module.
  XtcIndexInputModule (const std::string& name) ;

  /// Method which is called once at the beginning of the job
  virtual void beginJob(Event& evt, Env& env);

  psana::Index& index() {return _idx;}

  // Destructor
  virtual ~XtcIndexInputModule () ;

protected:
  
private:
  std::queue<DgramPieces> _queue;
  Index _idx;
};

} // namespace PSXtcInput

#endif // PSXTCINPUT_XTCINDEXINPUTMODULE_H
