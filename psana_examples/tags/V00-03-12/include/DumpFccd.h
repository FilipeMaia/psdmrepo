#ifndef PSANA_EXAMPLES_DUMPFCCD_H
#define PSANA_EXAMPLES_DUMPFCCD_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpFccd.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace psana_examples {

/**
 *  @brief Example module class for psana
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

class DumpFccd : public Module {
public:

  // Default constructor
  DumpFccd (const std::string& name) ;

  // Destructor
  virtual ~DumpFccd () ;
  
  /// Method which is called at the beginning of the calibration cycle
  virtual void beginCalibCycle(Event& evt, Env& env);
  
  /// Method which is called with event data
  virtual void event(Event& evt, Env& env);
  
protected:

private:

  Source m_src;

};

} // namespace psana_examples

#endif // PSANA_EXAMPLES_DUMPFCCD_H
