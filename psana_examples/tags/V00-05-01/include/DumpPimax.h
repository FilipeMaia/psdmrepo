#ifndef PSANA_EXAMPLES_DUMPPIMAX_H
#define PSANA_EXAMPLES_DUMPPIMAX_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: DumpPimax.h 1919 2014-03-17 09:25:00Z dubrovin@SLAC.STANFORD.EDU $
//
// Description:
//	Class DumpPimax.
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
 *  @version $Id: DumpPimax.h 1919 2014-03-17 09:25:00Z dubrovin@SLAC.STANFORD.EDU $
 *
 *  @author Mikhail Dubrovin
 */

class DumpPimax : public Module {
public:

  // Default constructor
  DumpPimax (const std::string& name) ;

  // Destructor
  virtual ~DumpPimax () ;

  /// Method which is called at the beginning of the calibration cycle
  virtual void beginCalibCycle(Event& evt, Env& env);
  
  /// Method which is called with event data
  virtual void event(Event& evt, Env& env);
  
protected:

private:

  Source m_src;

};

} // namespace psana_examples

#endif // PSANA_EXAMPLES_DUMPPIMAX_H
