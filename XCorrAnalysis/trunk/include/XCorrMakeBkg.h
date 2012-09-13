#ifndef PSANA_SXR61612_XCORRTIMINGTOOL_H
#define PSANA_SXR61612_XCORRTIMINGTOOL_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XCorrMakeBkg.
//
// Author: Sanne de Jong
//         adapted for psana by Ingrid Ofte
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
//#include "CCheader.h"
//#include "CCheader_array.h"

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
 *  @author Sanne de Jong
 */

class XCorrMakeBkg : public Module {
public:

  // Default constructor
  XCorrMakeBkg (const std::string& name) ;

  // Destructor
  virtual ~XCorrMakeBkg() ;

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
  int m_runNr;

  Source m_opalSrc;    // Opal camera source
  std::string m_darkFileOut;

  shared_ptr< ndarray<uint16_t, 2> > m_ccaverage;
  //ndarray<uint16_t,2> *m_ccaverage;
  
};

} // namespace XCorrAnalysis

#endif // PSANA_SXR61612_XCORRTIMINGTOOL_H
