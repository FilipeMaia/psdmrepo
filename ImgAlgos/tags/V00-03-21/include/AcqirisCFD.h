#ifndef IMGALGOS_ACQIRISCFD_H
#define IMGALGOS_PKG_ACQIRISCFD_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisCFD.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"

class TH1F;

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgAlgos {

/// @addtogroup ImgAlgos

/**
 *  @ingroup ImgAlgos
 *
 *  @brief Example module class for psana
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Christopher Ogrady
 */

class AcqirisCFD : public Module {
public:

  // Default constructor
  AcqirisCFD (const std::string& name) ;

  // Destructor
  virtual ~AcqirisCFD () ;

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

  /// Source address of the data object
  Pds::Src        m_src;

  /// String with source name
  Source          m_str_src;

  /// String with key for input data
  std::string     m_key_wform;
  std::string     m_key_wtime;
  std::string     m_key_edges;

  std::string m_baselines;
  std::string m_thresholds;
  std::string m_fractions;
  std::string m_deadtimes;
  std::string m_leading_edges;

  std::vector<double> v_baseline;
  std::vector<double> v_threshold;
  std::vector<double> v_fraction;
  std::vector<double> v_deadtime;
  std::vector<bool>   v_leading_edge;
  unsigned _evtcount;
};

} // namespace ImgAlgos

#endif // IMGALGOS_ACQIRISCFD_H
