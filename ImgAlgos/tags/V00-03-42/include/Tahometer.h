#ifndef IMGALGOS_TAHOMETER_H
#define IMGALGOS_TAHOMETER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Tahometer.
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
#include "ImgAlgos/TimeInterval.h"

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
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class Tahometer : public Module {
public:

  typedef double amplitude_t; 

  // Default constructor
  Tahometer (const std::string& name) ;

  // Destructor
  virtual ~Tahometer () ;

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
  void     printInputParameters();
  void     procEvent(Event& evt, Env& env);
  void     printTimeIntervalSummary(Event& evt, double dt_sec, long counter);
  void     printSummaryForParser(Event& evt, double dt_sec, long counter);

private:

  unsigned    m_print_bits;

  long        m_dn;
  long        m_count_dn;
  long        m_count;

  std::string m_str_runnum;

  TimeInterval *m_time;
  TimeInterval *m_time_dn;
};

} // namespace ImgAlgos

#endif // IMGALGOS_TAHOMETER_H
