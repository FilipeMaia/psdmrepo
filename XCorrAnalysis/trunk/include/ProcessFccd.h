#ifndef PSANA_SXR61612_PROCESSFCCD_H
#define PSANA_SXR61612_PROCESSFCCD_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ProcessFccd.
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

class ProcessFccd : public Module {
public:

  // Default constructor
  ProcessFccd (const std::string& name) ;

  // Destructor
  virtual ~ProcessFccd () ;

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

  void QuickTest(Event& evt, Env& env);

protected:

private:

  // Data members, this is for example purposes only
  long m_count;
  Source m_src;     // Detector address (can be set in config file)
  std::string m_img_out;// Name in the event of output image
  int m_BackgroundParam;
  int m_pkiRegionStart, m_pkiRegionEnd; // Peak Intensity Region
  int m_bgiRegionStart, m_bgiRegionEnd; // Background Intensity Region


  // Make array to store a copy of the event's image
  uint16_t RawImage[500][576];
  ndarray<uint16_t, 2> m_raw_image; // ndarray wrapper for the image

  typedef float WholeImage[480][480];
  typedef float HalfImage[240][480];
  
  WholeImage whole_image;
  HalfImage half_image;
  WholeImage* image_array;
  

  };// class ProcessFccd

} // namespace XCorrAnalysis

#endif // PSANA_SXR61612_PROCESSFCCD_H
