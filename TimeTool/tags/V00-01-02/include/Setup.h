#ifndef PSANA_TIMETOOL_SETUP_H
#define PSANA_TIMETOOL_SETUP_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimeTool::Setup.
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
#include <vector>

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include <string>

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace TimeTool {

/// @addtogroup TimeTool

/**
 *  @ingroup TimeTool
 *
 *  @brief Setup computes the autocorrelation function of several
 *  reference images for use in creating the digital filter weights
 *  for the timetool Analyze module.  The autocorrelation function
 *  is used to generate the inverted "noise" matrix against which the 
 *  template signal vector is multiplied to generate the weights vector.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Matthew J. Weaver
 */

class Setup : public Module {
public:

  // Default constructor
  Setup (const std::string& name) ;

  // Destructor
  virtual ~Setup () ;

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

  Source      m_get_key;  // Key for retrieving camera image
  
  int m_eventcode_nobeam; // use this eventcode for detecting no beam
  int m_eventcode_skip  ; // use this eventcode for skipping (no laser)

  std::string m_ipm_get_key;  // use this ipm threshold for detecting no beam
  double      m_ipm_beam_threshold;

  bool     m_projectX ;  // project image onto X axis
  int      m_proj_cut ;  // valid projection must be at least this large

  unsigned m_sig_roi_lo[2];  // image sideband is projected within ROI
  unsigned m_sig_roi_hi[2];  // image sideband is projected within ROI

  unsigned m_sb_roi_lo[2];  // image sideband is projected within ROI
  unsigned m_sb_roi_hi[2];  // image sideband is projected within ROI

  unsigned m_frame_roi[2];  // frame data is an ROI

  double   m_sb_avg_fraction ; // rolling average fraction (1/N)
  double   m_ref_avg_fraction; // rolling average fraction (1/N)

  ndarray<double,1> m_ref;     // accumulated reference
  ndarray<double,1> m_sb_avg;  // averaged sideband

  std::string m_ref_store; // store reference to text file

  unsigned m_pedestal;

  class DumpH {
  public:
    DumpH() : hraw(0), hrat(0), hacf(0) {}
    PSHist::H1* hraw;
    PSHist::H1* hrat;
    PSHist::H1* hacf;
  };
  std::vector<DumpH> m_hdump;

  PSHist::Profile* m_hacf;

  unsigned m_count;
  };
} // namespace TimeTool

#endif // PSANA_TIMETOOL_SETUP_H
