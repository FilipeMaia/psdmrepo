#ifndef IMGALGOS_IMGCALIB_H
#define IMGALGOS_IMGCALIB_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgCalib.
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
#include "ImgAlgos/ImgParametersV1.h"

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

class ImgCalib : public Module {
public:

  // Default constructor
  ImgCalib (const std::string& name) ;

  // Destructor
  virtual ~ImgCalib () ;

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
  void printEventRecord(Event& evt);
  void defineImageShape(Event& evt);

protected:
  void init(Event& evt, Env& env);
  void saveImageInEvent(Event& evt, double *p_data=0, const unsigned *shape=0);
private:

  Pds::Src        m_src;              // source address of the data object
  std::string     m_str_src;          // string with source name
  std::string     m_key_in;           // string with key for input data
  std::string     m_key_out;          // string with key for output image
  std::string     m_fname_peds;       // string file name for pedestals 
  std::string     m_fname_mask;       // string file name for mask
  std::string     m_fname_bkgd;       // string file name for background
  std::string     m_fname_gain;       // string file name for gain factors     
  unsigned        m_print_bits;       // bit mask for print options
  long            m_count;            // local event counter

  unsigned        m_shape[2];         // image shape
  unsigned        m_size;             // image size (number of elements)
    
  ImgParametersV1* m_peds;
  ImgParametersV1* m_mask;
  ImgParametersV1* m_bkgd;
  ImgParametersV1* m_gain;
};

} // namespace ImgAlgos

#endif // IMGALGOS_IMGCALIB_H
