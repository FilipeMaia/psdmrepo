#ifndef CSPADIMAGE_CSPADTEST_H
#define CSPADIMAGE_CSPADTEST_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadTest.
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

namespace CSPadImage {

/**
 *  @brief Example module class for psana
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class CSPadTest : public Module {
public:

  // Default constructor
  CSPadTest (const std::string& name) ;

  // Destructor
  virtual ~CSPadTest () ;

  /// Method which is called once at the beginning of the job
  //virtual void beginJob(Event& evt, Env& env);
  
  /// Method which is called at the beginning of the run
  virtual void beginRun(Event& evt, Env& env);
  
  /// Method which is called at the beginning of the calibration cycle
  //virtual void beginCalibCycle(Event& evt, Env& env);
  
  /// Method which is called with event data, this is the only required 
  /// method, all other methods are optional
  virtual void event(Event& evt, Env& env);
  
  /// Method which is called at the end of the calibration cycle
  //virtual void endCalibCycle(Event& evt, Env& env);

  /// Method which is called at the end of the run
  //virtual void endRun(Event& evt, Env& env);

  /// Method which is called once at the end of the job
  virtual void endJob(Event& evt, Env& env);

  //void iterateOverData(const uint16_t* data);
  void iterateOverData(const uint16_t data[][388]);

protected:

private:

  // Data members, this is for example purposes only
  
  Source m_src;         // Data source set from config file
  unsigned m_maxEvents;
  bool m_filter;
  long m_count;

  int m_Nquads;
  int m_Nrows;
  int m_Ncols;


};

} // namespace CSPadImage

#endif // CSPADIMAGE_CSPADTEST_H
