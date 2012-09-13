#ifndef PSANA_SXR61612_PPWF_XCORR_H
#define PSANA_SXR61612_PPWF_XCORR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ppwf_xcorr.
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

namespace XCorrAnalysis {

/// @addtogroup XCorrAnalysis

/**
 *  @ingroup XCorrAnalysis
 *
 *  @brief Example module class for psana
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Ingrid Ofte
 */

class ppwf_xcorr : public Module {
public:

  // Default constructor
  ppwf_xcorr (const std::string& name) ;

  // Destructor
  virtual ~ppwf_xcorr () ;

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

  // Data members, this is for example purposes only
  
  Source m_evrSrc;     // Evr address (can be set in config file)
  Source m_encSrc;     // Encoder address (can be set in config file)
  Source m_pcSrc;      // PhaseCavity address (can be set in config file)
  Source m_ipmSrc;     // IPIMB address
  Source m_fccdSrc;    // FCCD camera source
  Source m_opalSrc;    // Opal camera source
  Source m_acqSrc;     // Waveform source
  unsigned m_maxEvents;
  bool m_filter;
  long m_count;

};

} // namespace XCorrAnalysis

#endif // PSANA_SXR61612_PPWF_XCORR_H
