#ifndef PSANA_SXR61612_DELAYTIMEBIN_H
#define PSANA_SXR61612_DELAYTIMEBIN_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DelayTimeBin
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

#define MaxEvents 1000000

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

class DelayTimeBin : public Module {
public:

  // Default constructor
  DelayTimeBin (const std::string& name) ;

  // Destructor
  virtual ~DelayTimeBin () ;

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

  // Data members
  long m_count;
  int m_encoder_channel;
  int m_nBadDelayTimes;

  // Arrays 
  double m_EncoderArray[MaxEvents];
  double m_PhaseCavityArray[4][MaxEvents];
  float  m_DelayTimeArray[MaxEvents];


  // parameters
  Pds::Src     m_encSrc;     // Encoder address (can be set in config file)
  Pds::Src     m_pcSrc;      // PhaseCavity address (can be set in config file)
  std::string  m_xcorrSrc;   // Cross-correlator time name in the event. 

  // Encoder parameters, for converting position to picoseconds
  float m_Delay_a;
  float m_Delay_b;
  float m_Delay_c;
  float m_Delay_0;

  // Binning parameters
  int m_NumberBins;
  float m_StartTime;
  float m_EndTime;
  double m_DeltaTime;

  // This parameter determines method for jitter correction.
  int m_DelayTimeFlag;
  int m_MaxEvents;

}; // class DelayTimeBin

} // namespace XCorrAnalysis

#endif // PSANA_SXR61612_DELAYTIMEBIN_H
