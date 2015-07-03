#ifndef PSANA_EXAMPLES_DUMPGSC16AI_H
#define PSANA_EXAMPLES_DUMPGSC16AI_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpGsc16ai.
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

/// @addtogroup psana_examples

/**
 *  @ingroup psana_examples
 *
 *  @brief Example module class for psana
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Andy Salnikov
 */

class DumpGsc16ai : public Module {
public:

  // Default constructor
  DumpGsc16ai (const std::string& name) ;

  // Destructor
  virtual ~DumpGsc16ai () ;

  /// Method which is called at the beginning of the calibration cycle
  virtual void beginCalibCycle(Event& evt, Env& env);
  
  /// Method which is called with event data, this is the only required 
  /// method, all other methods are optional
  virtual void event(Event& evt, Env& env);
  
protected:

private:

  Source m_src;         // Data source set from config file

};

} // namespace psana_examples

#endif // PSANA_EXAMPLES_DUMPGSC16AI_H
