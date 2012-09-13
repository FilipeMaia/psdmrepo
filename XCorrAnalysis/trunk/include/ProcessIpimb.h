#ifndef PSANA_SXR61612_PROCESSIPIMB_H
#define PSANA_SXR61612_PROCESSIPIMB_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ProcessIpimb.
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
#define MaxEvents 1000000

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

class ProcessIpimb : public Module {
public:

  // Default constructor
  ProcessIpimb (const std::string& name) ;

  // Destructor
  virtual ~ProcessIpimb () ;

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
  long m_count;
  Source m_ipmSrc;     // IPIMB address
  int m_ipmI0; // channel number used for I0
  double m_ipmOffset;
  int m_NormalizationFlag;
  double m_IpimbLowerThreshold;
  double m_IpimbUpperThreshold;

  // Arrays 
  double m_IpimbArray[4][MaxEvents];


  char m_filename[256];
  std::string m_OutLabel;
};

} // namespace XCorrAnalysis

#endif // PSANA_SXR61612_PROCESSIPIMB_H
