#ifndef PSANA_EXAMPLES_DUMPCAMERA_H
#define PSANA_EXAMPLES_DUMPCAMERA_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpCamera.
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

class DumpCamera : public Module {
public:

  // Default constructor
  DumpCamera (const std::string& name) ;

  // Destructor
  virtual ~DumpCamera () ;
  
  /// Method which is called at the beginning of the calibration cycle
  virtual void beginCalibCycle(Env& env);
  
  /// Method which is called with event data, this is the only required 
  /// method, all other methods are optional
  virtual void event(Event& evt, Env& env);
  
protected:

private:

  // Data members, this is for example purposes only
  Source m_camSrc;

};

} // namespace psana_examples

#endif // PSANA_EXAMPLES_DUMPCAMERA_H
