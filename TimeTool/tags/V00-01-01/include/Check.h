#ifndef PSANA_TIMETOOL_CHECK_H
#define PSANA_TIMETOOL_CHECK_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimeTool::Check.
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
 *  @brief Module to check the result of the TimeTool::Analyze module.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Matthew J. Weaver
 */

class Check : public Module {
public:

  // Default constructor
  Check (const std::string& name) ;

  // Destructor
  virtual ~Check () ;

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

  std::string m_get_key1;  // Key for retrieving data
  std::string m_get_key2;  // Key for retrieving data

  std::string m_angle_shift;
  double      m_angle_shift_offset;  

  std::vector<double> m_phcav1_limits;
  std::vector<double> m_phcav2_limits;
  std::vector<double> m_tt_calib;

  std::vector<double> m_amplitude_binning;
  std::vector<double> m_position_binning;
  std::vector<double> m_width_binning;

  PSHist::H1*      m_ampl;
  PSHist::H1*      m_fltp;
  PSHist::H1*      m_fltw;
  PSHist::H2*      m_ampl_v_fltp;
  PSHist::H1*      m_namp;
  PSHist::H2*      m_namp2;

  PSHist::Profile* m_p1_v_p2;
  PSHist::Profile* m_pos_v_p1;
  PSHist::Profile* m_pos_v_p2;
  PSHist::Profile* m_tt_v_p1;
  PSHist::Profile* m_tt_v_p2;
  PSHist::H1*      m_p1_m_p2;
  PSHist::H1*      m_tt_m_p1;
  PSHist::H1*      m_tt_m_p2;
  PSHist::H2*      m_tt_v_p2_2d;
  };
} // namespace TimeTool

#endif // PSANA_TIMETOOL_CHECK_H
