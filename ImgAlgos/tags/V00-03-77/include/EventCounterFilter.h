#ifndef IMGALGOS_EVENTCOUNTERFILTER_H
#define IMGALGOS_EVENTCOUNTERFILTER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EventCounterFilter.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <stdint.h> // uint8_t, uint32_t, etc.
#include <iomanip>  // for setw, setfill

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

//----------------

namespace ImgAlgos {


class EventCounterFilter : public Module {
public:

  // Default constructor
  EventCounterFilter (const std::string& name) ;

  // Destructor
  virtual ~EventCounterFilter () ;

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
  int      m_mode;
  std::string m_ifname;
  unsigned m_print_bits;
  unsigned m_count_evt;
  unsigned m_count_sel;
 
  std::vector<unsigned> v_evnum;
  std::vector<unsigned>::const_iterator v_evnum_iter;

  void loadFile();
  void printEventVector();
  void printInputParameters();
  bool eventIsSelected(Event& evt, Env& env);
};

} // namespace ImgAlgos

#endif // IMGALGOS_EVENTCOUNTERFILTER_H
