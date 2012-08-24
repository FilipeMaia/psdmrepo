#ifndef IMGALGOS_IMGCALIB_H
#define IMGALGOS_IMGCALIB_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgCalib.
//      Apply corrections to 2d image using pedestals, background, gain factor, and mask.
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
 *  Apply corrections to 2d image using pedestals, background, gain factor, and mask.
 *
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
  void defImgIndexesForBkgdNorm();
  void procEvent(Event& evt, Env& env);
  void normBkgd();
  void saveImageInEvent(Event& evt);

private:

  Pds::Src        m_src;              // source address of the data object
  std::string     m_str_src;          // string with source name
  std::string     m_key_in;           // string with key for input data
  std::string     m_key_out;          // string with key for output image
  std::string     m_fname_peds;       // string file name for pedestals 
  std::string     m_fname_bkgd;       // string file name for background
  std::string     m_fname_gain;       // string file name for gain factors     
  std::string     m_fname_mask;       // string file name for mask
  double          m_mask_val;         // Value substituted for masked bits
  unsigned        m_row_min;          // window for background normalization
  unsigned        m_row_max;          // window for background normalization
  unsigned        m_col_min;          // window for background normalization
  unsigned        m_col_max;          // window for background normalization
  unsigned        m_print_bits;       // bit mask for print options
  long            m_count;            // local event counter

  unsigned        m_shape[2];         // image shape
  unsigned        m_cols;             // number of columns in the image 
  unsigned        m_rows;             // number of rows    in the image 
  unsigned        m_size;             // image size = m_cols * m_rows (number of elements)

  bool m_do_peds;                     // flag: true = do pedestal subtraction
  bool m_do_mask;                     // flag: true = apply mask
  bool m_do_bkgd;                     // flag: true = subtract background
  bool m_do_gain;                     // flag: true = apply the gain correction

  ImgParametersV1* m_peds;
  ImgParametersV1* m_bkgd;
  ImgParametersV1* m_gain;
  ImgParametersV1* m_mask;

  ImgParametersV1::pars_t* m_peds_data;
  ImgParametersV1::pars_t* m_bkgd_data;
  ImgParametersV1::pars_t* m_gain_data;
  ImgParametersV1::pars_t* m_mask_data;

  std::vector<unsigned> v_inds;       // vector of the image indexes for background normalization

  double           m_norm;            // Normalization factor for background subtraction

  const double*    m_rdat;            // Raw image data 
  double*          m_cdat;            // Calibrated data for image
};

} // namespace ImgAlgos

#endif // IMGALGOS_IMGCALIB_H
