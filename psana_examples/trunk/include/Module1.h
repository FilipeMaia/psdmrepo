#ifndef PSANA_EXAMPLES_MODULE1_H
#define PSANA_EXAMPLES_MODULE1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Module1.
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
 *  @brief Example of the user analysis module.
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

class Module1 : public Module {
public:

  // Default constructor
  Module1 (const std::string& name) ;

  // Destructor
  virtual ~Module1 () ;

  /// Method which is called with event data
  virtual void event(Event& evt, Env& env);
  
protected:

private:

  // Data members
  unsigned m_count;
  unsigned m_maxEvents;
  bool m_filter;
  
};

} // namespace psana_examples

#endif // PSANA_EXAMPLES_MODULE1_H
