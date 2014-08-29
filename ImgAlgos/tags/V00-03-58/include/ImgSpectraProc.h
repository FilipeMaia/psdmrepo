#ifndef IMGALGOS_IMGSPECTRAPROC_H
#define IMGALGOS_IMGSPECTRAPROC_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgSpectraProc.
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
#include "PSEvt/Source.h"
#include "MsgLogger/MsgLogger.h"

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
 *  @brief ImgSpectraProc show an example of how to get spectra after ImgSpectra
 *
 *  ImgSpectraProc psana module:
 *  - gets the ndarray<double,2> object with spectra from event,
 *  - for example, prints extracted spectra
 * 
 *  This software was developed for the LCLS project. If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see CSPadImageProducer
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class ImgSpectraProc : public Module {
public:

  // Default constructor
  ImgSpectraProc (const std::string& name) ;

  // Destructor
  virtual ~ImgSpectraProc () ;

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

  void printInputParameters();
  void printEventRecord(Event& evt, std::string=std::string());
  void getSpectra(Event& evt, bool print_msg = false);
  void printSpectra(Event& evt);

private:

  Pds::Src    m_src;
  Source      m_str_src;     // i.e. Opal:
  std::string m_key_in;      // input key
  unsigned    m_print_bits;
  long        m_count;

  unsigned      m_rows;
  unsigned      m_cols;
  const double* m_data;
  unsigned      m_shape[2];

protected:

//--------------------
//--------------------
//--------------------
//--------------------
};

} // namespace ImgAlgos

#endif // IMGALGOS_IMGSPECTRAPROC_H
