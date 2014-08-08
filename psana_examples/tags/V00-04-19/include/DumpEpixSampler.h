#ifndef PSANA_EXAMPLES_DUMPEPIXSAMPLER_H
#define PSANA_EXAMPLES_DUMPEPIXSAMPLER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpEpixSampler.
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
 *  @brief Example psana module to dump EpixSampler data
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Andy Salnikov
 */

class DumpEpixSampler : public Module {
public:

  // Default constructor
  DumpEpixSampler (const std::string& name) ;

  // Destructor
  virtual ~DumpEpixSampler () ;

  /// Method which is called at the beginning of the run
  virtual void beginRun(Event& evt, Env& env);
  
  /// Method which is called with event data, this is the only required 
  /// method, all other methods are optional
  virtual void event(Event& evt, Env& env);
  
protected:

private:

  // Data members, this is for example purposes only
  
  Source m_src;         // Data source set from config file

};

} // namespace psana_examples

#endif // PSANA_EXAMPLES_DUMPEPIXSAMPLER_H
