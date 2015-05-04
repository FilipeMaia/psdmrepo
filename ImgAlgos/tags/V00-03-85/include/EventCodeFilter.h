#ifndef IMGALGOS_EVENTCODEFILTER_H
#define IMGALGOS_EVENTCODEFILTER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EventCodeFilter.
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


class EventCodeFilter : public Module {
public:

  // Default constructor
  EventCodeFilter (const std::string& name) ;

  // Destructor
  virtual ~EventCodeFilter () ;

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
  Source   m_source;
  //std::string m_str_evcodes;
  uint32_t m_evcode;
  int      m_mode;
  unsigned m_print_bits;
  unsigned m_count_evt;
  unsigned m_count_sel;
 
  //std::vector<int> v_evcode;

  //void convertStringEvtCodesToVector();
  //void printEventCodeVector();
  void printFIFOEventsInEvent(Event& evt);
  void printInputParameters();
  bool eventIsSelected(Event& evt, Env& env);
  bool evcodeIsAvailable(Event& evt, Env& env);

};

} // namespace ImgAlgos

#endif // IMGALGOS_EVENTCODEFILTER_H
