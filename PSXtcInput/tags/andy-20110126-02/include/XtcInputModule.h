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

//----------------------
// Base Class Headers --
//----------------------
#include "psana/InputModule.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PsXtcInput {

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

private:

  // Data members

};

} // namespace PsXtcInput

#endif // PSXTCINPUT_XTCINPUTMODULE_H
