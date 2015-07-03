#ifndef PSANA_EXAMPLES_DUMPPARTITION_H
#define PSANA_EXAMPLES_DUMPPARTITION_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id:
//
// Description:
//	Class DumpPartition.
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
 *  @version $Id:
 *
 */

class DumpPartition : public Module {
public:

  // Default constructor
  DumpPartition (const std::string& name) ;

  // Destructor
  virtual ~DumpPartition () ;

  /// Method which is called at the beginning of the job
  virtual void beginJob(Event& evt, Env& env);

  /// Method which is called with event data
  virtual void event(Event& evt, Env& env);
  
protected:

private:
};

} // namespace psana_examples

#endif // PSANA_EXAMPLES_DUMPPARTITION_H
