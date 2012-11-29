#ifndef PSANA_EXAMPLES_PNCCDIMAGEPRODUCER_H
#define PSANA_EXAMPLES_PNCCDIMAGEPRODUCER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: PnccdImageProducer.h 0001 2012-07-06 09:00:00Z dubrovin@SLAC.STANFORD.EDU $
//
// Description:
//	Class PnccdImageProducer.
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

namespace ImgAlgos {

/**
 *  @brief Example module class for psana
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id: PnccdImageProducer.h 0001 2012-07-06 09:00:00Z dubrovin@SLAC.STANFORD.EDU $
 *
 *  @author Mikhail Dubrovin
 */

class PnccdImageProducer : public Module {
public:

  // Default constructor
  PnccdImageProducer (const std::string& name) ;

  // Destructor
  virtual ~PnccdImageProducer () ;

  /// Method which is called once at the beginning of the job
  virtual void beginJob(Event& evt, Env& env);

  /// Method which is called at the beginning of the calibration cycle
  virtual void beginCalibCycle(Event& evt, Env& env);
  
  /// Method which is called with event data
  virtual void event(Event& evt, Env& env);
  
protected:

  void printInputParameters();
 
private:

  Pds::Src    m_src;
  Source      m_str_src;
  std::string m_key_in; 
  std::string m_key_out;
  unsigned    m_print_bits;
};

} // namespace ImgAlgos

#endif // PSANA_EXAMPLES_PNCCDIMAGEPRODUCER_H
