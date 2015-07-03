#ifndef PSANA_EXAMPLES_DUMPL3T_H
#define PSANA_EXAMPLES_DUMPL3T_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpL3T.
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
 *  Psana module which dumps L3T information.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Andy Salnikov
 */

class DumpL3T : public Module {
public:

  // Default constructor
  DumpL3T (const std::string& name) ;

  // Destructor
  virtual ~DumpL3T () ;

  /// Method which is called at the beginning of the run
  virtual void beginRun(Event& evt, Env& env);
  
  /// Method which is called with event data, this is the only required 
  /// method, all other methods are optional
  virtual void event(Event& evt, Env& env);
  
protected:

private:

  Source m_src;         // Data source set from config file

};

} // namespace psana_examples

#endif // PSANA_EXAMPLES_DUMPL3T_H
