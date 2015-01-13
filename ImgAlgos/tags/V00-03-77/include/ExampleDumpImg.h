#ifndef IMGALGOS_EXAMPLEDUMPIMG_H
#define IMGALGOS_EXAMPLEDUMPIMG_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ExampleDumpImg.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <stdint.h>

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

/// @addtogroup ImgAlgos

/**
 *  @ingroup ImgAlgos
 *
 *  @brief Example module which gets 2-d image from the evt store as ndarray<const data_t, 2> and print its part.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class ExampleDumpImg : public Module {
public:

  typedef float data_t;

  // Default constructor
  ExampleDumpImg (const std::string& name) ;

  // Destructor
  virtual ~ExampleDumpImg () ;

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

  void printInputParameters();
  void procEvent(Event& evt);

private:

  template <typename T> 
    bool procEventForType(Event& evt);
 
  Pds::Src    m_src;        // source address of the data object 
  Source      m_source;     // Data source set from config file
  std::string m_key;        // String key for input data
  unsigned    m_print_bits; // Bit mask for print options
  unsigned    m_row_dump;   // Image row to dump
  long        m_count;      // Event counter
 
};

} // namespace ImgAlgos

#endif // IMGALGOS_EXAMPLEDUMPIMG_H
