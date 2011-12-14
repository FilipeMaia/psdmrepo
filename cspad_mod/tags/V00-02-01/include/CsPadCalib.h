#ifndef CSPAD_MOD_CSPADCALIB_H
#define CSPAD_MOD_CSPADCALIB_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadCalib.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <set>

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psddl_psana/cspad.ddl.h"

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
 *  @brief Module which performs CsPad calibration.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Andy Salnikov
 */

class CsPadCalib : public Module {
public:

  // Default constructor
  CsPadCalib (const std::string& name) ;

  // Destructor
  virtual ~CsPadCalib () ;

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

  // add calibration proxies for DataV* classes
  void addProxyV1(const PSEvt::EventKey& key, Event& evt, Env& env);
  void addProxyV2(const PSEvt::EventKey& key, Event& evt, Env& env);

  // add calibration proxies for MiniElementV* classes
  void addProxyMini(const PSEvt::EventKey& key, Event& evt, Env& env);

private:


  std::string m_inkey;     ///< event key for non-calibrated data, default is empty
  std::string m_outkey;    ///< event key for calibrated data, default is "calibrated"
  bool m_doPedestals;  ///< do pedestal subtraction if set
  bool m_doPixelStatus;  ///< use pixel status data if set
  bool m_doCommonMode;  ///< do common mode correction if set

  
};

} // namespace cspad_mod

#endif // CSPAD_MOD_CSPADCALIB_H
