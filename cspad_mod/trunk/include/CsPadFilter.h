#ifndef CSPAD_MOD_CSPADFILTER_H
#define CSPAD_MOD_CSPADFILTER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadFilter.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <list>

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

namespace cspad_mod {

/// @addtogroup cspad_mod

/**
 *  @ingroup cspad_mod
 *
 *  @brief Module which performs CsPad filtering.
 *
 *  This class defines psana module which perform event filtering
 *  based on the signal in CsPad images. If this module determines that
 *  CsPad does not have sufficient signal it forces psana to skip
 *  remaining modules for this event. It also skips event if there is no
 *  CsPad data at all, this behavior can be changed by setting
 *  configuration option "skipIfNoData" to "no" or "false".
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Andy Salnikov
 */

class CsPadFilter : public Module {
public:

  // Default constructor
  CsPadFilter (const std::string& name) ;

  // Destructor
  virtual ~CsPadFilter () ;

  /// Method which is called once at the beginning of the job
  virtual void beginJob(Event& evt, Env& env);
  
  /// Method which is called at the beginning of the run
  virtual void beginRun(Event& evt, Env& env);
  
  /// Method which is called at the beginning of the calibration cycle
  virtual void beginCalibCycle(Event& evt, Env& env);
  
  /// Method which is called with event data, this is the only required 
  /// method, all other methods are optional
  virtual void event(Event& evt, Env& env);
  
  /// Method which is called at the end of the calibration cycle
  virtual void endCalibCycle(Event& evt, Env& env);

  /// Method which is called at the end of the run
  virtual void endRun(Event& evt, Env& env);

  /// Method which is called once at the end of the job
  virtual void endJob(Event& evt, Env& env);

protected:

private:

  Source m_src; ///< Data source address
  std::string m_key; ///< event key for input data, default is empty
  bool m_skipIfNoData; ///< If true then even is filtered out if there is no CsPad data
  int m_mode;   ///< Filter mode, see pdscalibdata::CsPadFilterV1, if negative use calib file
  std::list<double> m_param;  ///< Filter parameters, only if m_mode is non-negative

};

} // namespace cspad_mod

#endif // CSPAD_MOD_CSPADFILTER_H
