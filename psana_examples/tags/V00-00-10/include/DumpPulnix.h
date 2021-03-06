#ifndef PSANA_EXAMPLES_DUMPPULNIX_H
#define PSANA_EXAMPLES_DUMPPULNIX_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpPulnix.
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

class DumpPulnix : public Module {
public:

  // Default constructor
  DumpPulnix (const std::string& name) ;

  // Destructor
  virtual ~DumpPulnix () ;

  /// Method which is called at the beginning of the calibration cycle
  virtual void beginCalibCycle(Env& env);
  
  /// Method which is called with event data, this is the only required 
  /// method, all other methods are optional
  virtual void event(Event& evt, Env& env);
  
protected:

private:

  Source m_src;

};

} // namespace psana_examples

#endif // PSANA_EXAMPLES_DUMPPULNIX_H
