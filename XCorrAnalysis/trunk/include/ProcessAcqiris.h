#ifndef PSANA_SXR61612_PROCESSACQIRIS_H
#define PSANA_SXR61612_PROCESSACQIRIS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ProcessAcqiris.
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
#define MaxSpectra 50000
#define NumberBins 300

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

class ProcessAcqiris : public Module {
public:

  // Default constructor
  ProcessAcqiris (const std::string& name) ;

  // Destructor
  virtual ~ProcessAcqiris () ;

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
  
  Pds::Src m_acqSrc;     // Waveform source
  long m_count;
  unsigned int m_nch;
  int m_DiodeStart;
  int m_DiodeLength;

  // Arrays
  double m_AcqDiodeSpect[20][MaxSpectra];
  double m_AcqDiodeArray[20][MaxEvents];
  double m_AcqDiodeScan[20][NumberBins];


};

} // namespace XCorrAnalysis

#endif // PSANA_SXR61612_PROCESSACQIRIS_H
