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
#include <vector>
#include <string>
#include <set>

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psddl_psana/timetool.ddl.h"
#include "TimeTool/EventDump.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

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

  /// helper function returns true if 'key' have value 'off'. Checks m_get_key source
  /// first, then no source. Throws fatal error if key is not found or doesn't have value
  /// 'on' or 'off'. Users moduleParameter to report errors
  bool getIsOffFromOnOffKey(const std::string & moduleParameter, const std::string & key, Event & evt);

  /// given a set of all the valid configuration keys, goes through the user specified 
  /// configuration keys and identifies invalid keys (to help find typos in config files)
  /// prints errors about invalid keys
  void checkForInvalidConfigKeys(const std::set<std::string> &validConfigKeys) const;

private:
  ndarray<unsigned,1> project(const ndarray<const uint16_t,2>&) const;

private:

  Source      m_get_key;  // Key for retrieving camera image
  std::string m_put_key;  // Key for inserting results into Event
  std::string m_beam_on_off_key; // Key for user to override beam logic
  std::string m_laser_on_off_key; // Key for user to override laser logic

  std::string m_ipm_get_key;  // use this ipm threshold for detecting no beam
  double      m_ipm_beam_threshold;

  //
  // The following parameters are optional.  They will default to
  // whatever configuration is found in the run data.
  //
  ndarray<const Psana::TimeTool::EventLogic,1> m_beam_logic;
  ndarray<const Psana::TimeTool::EventLogic,1> m_laser_logic;

  ndarray<const double,1> m_calib_poly; // polynomial coefficients converting to time

  bool        m_projectX_set;
  bool        m_projectX ;  // project image onto X axis
  int         m_proj_cut ;  // valid projection must be at least this large

  bool     m_sig_roi_set;
  unsigned m_sig_roi_lo[2];  // image signal is projected within ROI
  unsigned m_sig_roi_hi[2];  // image signal is projected within ROI

  bool     m_sb_roi_set;
  unsigned m_sb_roi_lo[2];  // image sideband is projected within ROI
  unsigned m_sb_roi_hi[2];  // image sideband is projected within ROI

  bool     m_ref_roi_set;
  unsigned m_ref_roi_lo[2];  // image reference is projected within ROI
  unsigned m_ref_roi_hi[2];  // image reference is projected within ROI

  unsigned m_frame_roi[2];  // frame data is an ROI

  double   m_sb_avg_fraction ; // rolling average fraction (1/N)
  double   m_ref_avg_fraction; // rolling average fraction (1/N)

  ndarray<const double,1> m_weights; // digital filter weights
  ndarray<double,1> m_ref_avg;       // accumulated reference
  ndarray<double,1> m_sb_avg;        // averaged sideband

  bool     m_analyze_projections;

  //
  //  These parameters are for persisting the accumulated reference
  //  and generating histograms
  //
  std::string m_ref_store; // store reference to text file

  bool m_use_calib_db_ref; // load initial reference from calibration;
  ndarray<double,2> m_ref_frame_avg; // initial pedestal loaded, then updated if plotting with eventdump
  bool setInitialReferenceIfUsingCalibirationDatabase(bool use_ref_roi, unsigned pdim);

  unsigned m_pedestal;

  class DumpH {
  public:
    DumpH() : hraw(0), href(0), hrat(0), hflt(0) {}
    PSHist::H1* hraw;
    PSHist::H1* href;
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

  boost::shared_ptr<PSHist::HManager> m_hmgr;
  TimeTool::EventDump m_eventDump;

  std::set<std::string> m_validConfigKeys;

  };
} // namespace TimeTool

#endif // PSANA_TIMETOOL_ANALYZE_H
