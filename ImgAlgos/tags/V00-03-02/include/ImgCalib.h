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
#include "ImgAlgos/GlobalMethods.h"

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

protected:
  void init(Event& evt, Env& env);
  void defImgIndexesForBkgdNorm();
  void procEvent(Event& evt, Env& env);
  void normBkgd();
  void saveImageInEvent(Event& evt);

private:

  Pds::Src        m_src;              // source address of the data object
  Source          m_str_src;          // string with source name
  std::string     m_key_in;           // string with key for input data
  std::string     m_key_out;          // string with key for output image
  std::string     m_fname_peds;       // string file name for pedestals 
  std::string     m_fname_bkgd;       // string file name for background
  std::string     m_fname_gain;       // string file name for gain factors     
  std::string     m_fname_mask;       // string file name for mask
  std::string     m_fname_nrms;       // string file name for threshold in nRMS
  double          m_mask_val;         // Value substituted for masked bits
  double          m_low_nrms;         // The low threshold number of RMS
  double          m_low_thre;         // The low threshold
  double          m_low_val;          // The value of substituting amplitude below threshold
  bool            m_do_thre;          // flag: true = apply the threshold
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
  bool m_do_nrms;                     // flag: true = apply the threshold as nRMS

  ImgParametersV1* m_peds;
  ImgParametersV1* m_bkgd;
  ImgParametersV1* m_gain;
  ImgParametersV1* m_mask;
  ImgParametersV1* m_nrms;

  ImgParametersV1::pars_t* m_peds_data;
  ImgParametersV1::pars_t* m_bkgd_data;
  ImgParametersV1::pars_t* m_gain_data;
  ImgParametersV1::pars_t* m_mask_data;
  ImgParametersV1::pars_t* m_nrms_data;

  std::vector<unsigned> v_inds;       // vector of the image indexes for background normalization

  double           m_norm;            // Normalization factor for background subtraction
  double*          m_cdat;            // Calibrated data for image

//-------------------

    template <typename T>
    bool procEventForType(Event& evt)
    {
     	shared_ptr< ndarray<T,2> > img = evt.get(m_str_src, m_key_in, &m_src);
     	if (img.get()) {

     	  const T* _rdat = img->data();

     	  // 1) Evaluate: m_cdat[i] = (_rdat[i] - m_peds[i] - m_norm*m_bkgd[i]) * m_gain[i]; 
     	  // 2) apply mask: m_mask[i];
     	  // 3) apply constant threshold: m_low_thre;
     	  // 4) apply nRMS threshold: m_low_nrms*m_nrms_data[i];

     	  //memcpy(m_cdat,m_rdat,m_size*sizeof(double)); 

	  ndarray<T,2> cdata(m_shape);
	  T* m_cdat  = cdata.data();

     	  for(unsigned i=0; i<m_size; i++) m_cdat[i] = (T)_rdat[i];

     	  if (m_do_peds) {             for(unsigned i=0; i<m_size; i++) m_cdat[i] -= (T) m_peds_data[i];         }
     	  if (m_do_bkgd) { normBkgd(); for(unsigned i=0; i<m_size; i++) m_cdat[i] -= (T)(m_bkgd_data[i]*m_norm); }
     	  if (m_do_gain) {             for(unsigned i=0; i<m_size; i++) m_cdat[i] *= (T) m_gain_data[i];         }

     	  if (m_do_mask) {             
	    T mask_val = (T)m_mask_val; 
     	    for(unsigned i=0; i<m_size; i++) {
     	      if (m_mask_data[i]==0) m_cdat[i] = mask_val; 
     	    }
     	  }

     	  if (m_do_thre) {             
	    T low_val  = (T) m_low_val;
	    T low_thre = (T) m_low_thre;
     	    for(unsigned i=0; i<m_size; i++) {
     	      if (m_cdat[i] < low_thre) m_cdat[i] = low_val; 
     	    }
     	  }

     	  if (m_do_nrms) {             
	    T low_val = (T) m_low_val;
     	    for(unsigned i=0; i<m_size; i++) {
     	      if (m_cdat[i] < (T) m_nrms_data[i]) m_cdat[i] = low_val; 
     	    }
     	  }

          save2DArrayInEvent<T> (evt, m_src, m_key_out, cdata);

     	  return true;
     	} 
        return false;
    }  

//-------------------

};

} // namespace ImgAlgos

#endif // IMGALGOS_IMGCALIB_H
