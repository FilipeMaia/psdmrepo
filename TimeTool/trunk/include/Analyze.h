#ifndef PSANA_TIMETOOL_ANALYZE_H
#define PSANA_TIMETOOL_ANALYZE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimeTool::Analyze.
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
 *  @brief The Analyze module projects an image within a region of interest
 *  and uses a digital filter to search for a characteristic edge shape
 *  associated with a relative time difference measurement.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Matthew J. Weaver
 */

class Analyze : public Module {
public:

  // Default constructor
  Analyze (const std::string& name) ;

  // Destructor
  virtual ~Analyze () ;

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
  ndarray<unsigned,1> project(const ndarray<const uint16_t,2>&) const;

private:

  Source      m_get_key;  // Key for retrieving camera image
  std::string m_put_key;  // Key for inserting results into Event
  
  bool m_use_online_config;   // use configuration parameters found in data

  int m_eventcode_nobeam; // use this eventcode for detecting no beam
  int m_eventcode_skip  ; // use this eventcode for skipping (no laser)

  std::string m_ipm_get_key;  // use this ipm threshold for detecting no beam
  double      m_ipm_beam_threshold;

  std::vector<double> m_calib_poly; // polynomial coefficients converting to time

  bool     m_projectX ;  // project image onto X axis
  int      m_proj_cut ;  // valid projection must be at least this large

  unsigned m_sig_roi_lo[2];  // image sideband is projected within ROI
  unsigned m_sig_roi_hi[2];  // image sideband is projected within ROI

  unsigned m_sb_roi_lo[2];  // image sideband is projected within ROI
  unsigned m_sb_roi_hi[2];  // image sideband is projected within ROI

  unsigned m_frame_roi[2];  // frame data is an ROI

  double   m_sb_avg_fraction ; // rolling average fraction (1/N)
  double   m_ref_avg_fraction; // rolling average fraction (1/N)

  ndarray<double,1> m_weights; // digital filter weights
  ndarray<double,1> m_ref;     // accumulated reference
  ndarray<double,1> m_sb_avg;  // averaged sideband

  std::string m_ref_store; // store reference to text file

  unsigned m_pedestal;

  class DumpH {
  public:
    DumpH() : hraw(0), hrat(0), hflt(0) {}
    PSHist::H1* hraw;
    PSHist::H1* hrat;
    PSHist::H1* hflt;
  };
  std::list<DumpH> m_hdump;

  ///  Use this feature to estimate the error by selecting signal
  ///  from one event and, in turn, applying each of the reference
  ///  images found to estimate the spread due to different reference
  ///  images.
  int               m_analyze_event;  // analyze the signal from one event
  ndarray<double,1> m_analyze_signal; // the signal to be applied to each new reference

  int m_count;
  };
} // namespace TimeTool

#endif // PSANA_TIMETOOL_ANALYZE_H
