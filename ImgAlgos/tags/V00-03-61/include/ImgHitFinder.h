#ifndef IMGALGOS_IMGHITFINDER_H
#define IMGALGOS_IMGHITFINDER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgHitFinder.
//      Apply corrections to 2d image using pedestals, mask, gain factor, and threshold
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

const unsigned k_val_def_u = 12345;
const double   k_val_def_d = 12345;


class ImgHitFinder : public Module {
public:

  typedef double data_out_t;
  typedef unsigned hitmap_t;

  // Default constructor
  ImgHitFinder (const std::string& name) ;

  // Destructor
  virtual ~ImgHitFinder () ;

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
  void procEvent(Event& evt, Env& env);
  void saveImageInEvent(Event& evt);

private:

  Pds::Src        m_src;              // source address of the data object
  Source          m_str_src;          // source 
  std::string     m_key_in;           // string with key for input data
  std::string     m_key_out;          // string with key for output image
  std::string     m_fname_peds;       // string file name for pedestals 
  std::string     m_fname_mask;       // string file name for mask
  std::string     m_fname_gain;       // string file name for gain factors     
  std::string     m_fname_thre;       // string file name for threshold
  double          m_masked_val;       // Value substituted for masked pixelss
  unsigned        m_thre_mode;        // Threshold mode
  double          m_thre_param;       // The low threshold
  double          m_thre_below_val;   // The value of substituting amplitude below threshold
  double          m_thre_above_val;   // The value of substituting amplitude above threshold (default=pixel intensity)
  unsigned        m_row_min;          // window roi
  unsigned        m_row_max;          // window roi
  unsigned        m_col_min;          // window roi
  unsigned        m_col_max;          // window roi
  unsigned        m_print_bits;       // bit mask for print options
  long            m_count;            // local event counter

  unsigned        m_shape[2];         // image shape
  unsigned        m_cols;             // number of columns in the image 
  unsigned        m_rows;             // number of rows    in the image 
  unsigned        m_size;             // image size = m_cols * m_rows (number of elements)

  bool m_do_peds;                     // flag: true = do pedestal subtraction
  bool m_do_mask;                     // flag: true = apply mask
  bool m_do_gain;                     // flag: true = apply the gain correction
  bool m_do_thre;                     // flag: true = apply the threshold

  // Objects with content loaded from files
  ImgParametersV1* m_peds;
  ImgParametersV1* m_mask;
  ImgParametersV1* m_gain;
  ImgParametersV1* m_thre;

  // Pointers to content of appropriate objects
  // ImgParametersV1::pars_t is float
  ImgParametersV1::pars_t* m_peds_data;
  ImgParametersV1::pars_t* m_mask_data;
  ImgParametersV1::pars_t* m_gain_data;
  ImgParametersV1::pars_t* m_thre_data;

  // ndarrays made of appropriate objects' content
  ndarray<ImgParametersV1::pars_t,2> m_peds_nda;
  ndarray<ImgParametersV1::pars_t,2> m_mask_nda;
  ndarray<ImgParametersV1::pars_t,2> m_gain_nda;
  ndarray<ImgParametersV1::pars_t,2> m_thre_nda;

//-------------------

  template <typename T, typename TOUT>
    bool procEventForType(Event& evt)
    {
     	shared_ptr< ndarray<const T,2> > img = evt.get(m_str_src, m_key_in, &m_src);
     	if (img.get()) {

     	  const T* _rdat = img->data();

     	  // 1) subtract pedestals
     	  // 2) apply gain factors
     	  // 3) apply mask
     	  // 4) apply threshold

	  //m_cdat = new double[m_size];
     	  //memcpy(m_cdat,m_rdat,m_size*sizeof(double)); 

	  ndarray<TOUT,2> cdata(m_shape);
	  TOUT* p_cdata  = cdata.data();
	  TOUT thre_below_val = (TOUT) m_thre_below_val; 
	  TOUT thre_above_val = (TOUT) m_thre_above_val; 

	  TOUT thre_param  = (TOUT) m_thre_param; 
	  TOUT masked_val  = (TOUT) m_masked_val; 


     	  for(unsigned i=0; i<m_size; i++) p_cdata[i] = (TOUT)_rdat[i]; // copy data image to array with output type

     	  if (m_do_peds) { for(unsigned i=0; i<m_size; i++) p_cdata[i] -= (TOUT) m_peds_data[i];         }

     	  if (m_do_gain) { for(unsigned i=0; i<m_size; i++) p_cdata[i] *= (TOUT) m_gain_data[i];         }

     	  if (m_do_mask) {             
     	    for(unsigned i=0; i<m_size; i++) {
     	      if (m_mask_data[i]==0) p_cdata[i] = masked_val; 
     	    }
     	  }


	  if (m_do_thre) {
               bool apply_above_thre_val = (m_thre_above_val == k_val_def_d) ? false : true; 

	       // Apply differend threshold mode
	       // --------------
	       //    thre_mode=1, Constant threshold, thre_param (for example=20) - threshold value in ADU
     	       if (m_thre_mode == 1) { 	       
     	         for(unsigned i=0; i<m_size; i++) {	       
     	           if (p_cdata[i] < thre_param) { p_cdata[i] = thre_below_val; continue; }             
	           if (apply_above_thre_val)      p_cdata[i] = thre_above_val;
     	         }
     	       }
	       	       
	       // --------------
	       //    thre_mode=2, Per-pixel threshold from file fname_thre multiplied by factor thre_param (for example=5)
     	       if (m_thre_mode == 2) {
     	         for(unsigned i=0; i<m_size; i++) {
     	           if (p_cdata[i] < (TOUT) m_thre_data[i]) p_cdata[i] = thre_below_val; continue;       
	           if (apply_above_thre_val)               p_cdata[i] = thre_above_val;
		 }
     	       }

	       // --------------

     	       if (m_thre_mode == 3) {

	           ndarray<hitmap_t, 2> hitmap(m_shape);
                   std::fill_n(hitmap.data(), int(m_size), hitmap_t(0));
                   find_hits<TOUT, hitmap_t>(cdata, m_thre_nda, hitmap);
                   save2DArrayInEvent<hitmap_t> (evt, m_src, m_key_out, hitmap);
	       }

	       // --------------

	  } // if (m_do_thre)

          if( m_print_bits & 32 && !m_count ) {
              if (m_thre_mode == 3) printInOutDataTypes<T,hitmap_t>(); 	  
	      else                  printInOutDataTypes<T,TOUT>(); 	
	  }

          if( m_print_bits &  8 ) MsgLog( name(), info, stringOf2DArrayData<T>(*img.get(), std::string("Raw img data: ")) );
          if( m_print_bits & 16 ) MsgLog( name(), info, stringOf2DArrayData<TOUT>(cdata, std::string("Proc img data:")) );

	  if (m_thre_mode != 3) save2DArrayInEvent<TOUT> (evt, m_src, m_key_out, cdata);

     	  return true;
     	} 
        return false;
    }  

//-------------------
  template <typename T, typename TOUT>
    void printInOutDataTypes()
    {
      MsgLog( name(), info,  "Data types: in=" << typeid(T).name()
                                    << " out=" << typeid(TOUT).name()
      );
    }

//-------------------

  template <typename TIN, typename TOUT>
  void find_hits ( const ndarray<TIN,2> input,
                   const ndarray<ImgParametersV1::pars_t,2>& threshold,
                   ndarray<TOUT,2> output)
  {
    for(unsigned j=m_row_min; j<m_row_max; j++)
      for(unsigned k=m_col_min; k<m_col_max; k++) {

         TIN v = input[j][k];
          if (v > threshold[j][k] &&
              v > input[j-1][k-1] &&
              v > input[j-1][k] &&
              v > input[j-1][k+1] &&
              v > input[j][k-1] &&
              v > input[j][k+1] &&
              v > input[j+1][k-1] &&
              v > input[j+1][k] &&
              v > input[j+1][k+1])
               output[j][k]=1;
	  else output[j][k]=0;
      }
  }

//-------------------
};

} // namespace ImgAlgos

#endif // IMGALGOS_IMGHITFINDER_H
