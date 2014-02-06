#ifndef IMGALGOS_NDARRCALIB_H
#define IMGALGOS_NDARRCALIB_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class NDArrCalib.
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
#include "ImgAlgos/GlobalMethods.h" // ::toString( const Pds::Src& src )
#include "PSCalib/PnccdCalibPars.h"
#include "PSCalib/CalibPars.h"
//#include "PSCalib/CSPad2x2CalibPars.h"

#include "psddl_psana/pnccd.ddl.h"

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

class NDArrCalib : public Module {
public:

  typedef double data_out_t;

  // Default constructor
  NDArrCalib (const std::string& name) ;

  // Destructor
  virtual ~NDArrCalib () ;

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
  void getCalibPars(Event& evt, Env& env);
  void getConfigPars(Env& env);
  void defImgIndexesForBkgdNorm();
  void initAtFirstGetNdarray(Event& evt, Env& env);
  void procEvent(Event& evt, Env& env);
  //void normBkgd();
  void saveImageInEvent(Event& evt);

private:

  Pds::Src        m_src;              // source address of the data object
  Source          m_str_src;          // string with source name
  std::string     m_key_in;           // string with key for input data
  std::string     m_key_out;          // string with key for output image
  bool            m_do_peds;          // flag: true = do pedestal subtraction
  bool            m_do_cmod;          // flag: true = do common mode subtraction
  bool            m_do_mask;          // flag: true = apply mask
  bool            m_do_bkgd;          // flag: true = subtract background
  bool            m_do_gain;          // flag: true = apply the gain correction
  bool            m_do_nrms;          // flag: true = apply the threshold as nRMS
  bool            m_do_thre;          // flag: true = apply the threshold
  std::string     m_fname_peds;       // string file name for pedestals 
  std::string     m_fname_bkgd;       // string file name for background
  std::string     m_fname_gain;       // string file name for gain factors     
  std::string     m_fname_mask;       // string file name for mask
  std::string     m_fname_nrms;       // string file name for threshold in nRMS
  double          m_mask_val;         // Value substituted for masked bits
  double          m_low_nrms;         // The low threshold number of RMS
  double          m_low_thre;         // The low threshold
  double          m_low_val;          // The value of substituting amplitude below threshold
  unsigned        m_row_min;          // window for background normalization
  unsigned        m_row_max;          // window for background normalization
  unsigned        m_col_min;          // window for background normalization
  unsigned        m_col_max;          // window for background normalization
  unsigned        m_print_bits;       // bit mask for print options
  long            m_count_event;      // local event counter
  long            m_count_get;        // local successful get() counter

  NDArrPars*      m_ndarr_pars;       // holds input data ndarray parameters
  unsigned        m_ndim;             // rank of the input data ndarray 
  unsigned        m_size;             // number of elements in the input data ndarray 
  DATA_TYPE       m_dtype;            // numerated datatype for data array

  unsigned        m_shape[2];         // image shape
  unsigned        m_cols;             // number of columns in the image 
  unsigned        m_rows;             // number of rows    in the image 


  PSCalib::CalibPars* m_calibpars;
  //PSCalib::PnccdCalibPars* m_calibpars;
  std::string m_typeGroup;            // for example: "PNCCD::CalibV1";

  data_out_t* p_cdata;                // pointer to calibrated data array

  const PSCalib::CalibPars::pedestals_t*     m_peds_data;
  const PSCalib::CalibPars::pixel_gain_t*    m_gain_data;
  const PSCalib::CalibPars::pixel_status_t*  m_mask_data;
  const PSCalib::CalibPars::common_mode_t*   m_cmod_data;

  PSCalib::CalibPars::pixel_bkgd_t*    m_bkgd_data;
  PSCalib::CalibPars::pixel_nrms_t*    m_nrms_data;

//   ImgParametersV1* m_peds;
//   ImgParametersV1* m_bkgd;
//   ImgParametersV1* m_gain;
//   ImgParametersV1* m_mask;
//   ImgParametersV1* m_nrms;

//   ImgParametersV1::pars_t* m_peds_data;
//   ImgParametersV1::pars_t* m_bkgd_data;
//   ImgParametersV1::pars_t* m_gain_data;
//   ImgParametersV1::pars_t* m_mask_data;
//   ImgParametersV1::pars_t* m_nrms_data;

  std::vector<unsigned> v_inds;       // vector of the image indexes for background normalization

  double           m_norm;            // Normalization factor for background subtraction
  //double*          m_cdat;            // Calibrated data for image

//-------------------

  template <typename T, typename TOUT>
    bool procEventForType(Event& evt)
    {
      if (m_ndim == 2) {
     	shared_ptr< ndarray<T,2> > shp2 = evt.get(m_str_src, m_key_in, &m_src);
     	if (shp2.get()) { applyCorrections<T,TOUT>(evt, shp2->data()); return true; } 
      }

      if (m_ndim == 3) {
     	shared_ptr< ndarray<T,3> > shp3 = evt.get(m_str_src, m_key_in, &m_src);
     	if (shp3.get()) { applyCorrections<T,TOUT>(evt, shp3->data()); return true; } 
      }

      if (m_ndim == 4) {
     	shared_ptr< ndarray<T,4> > shp4 = evt.get(m_str_src, m_key_in, &m_src);
     	if (shp4.get()) { applyCorrections<T,TOUT>(evt, shp4->data()); return true; } 
      }

      if (m_ndim == 5) {
     	shared_ptr< ndarray<T,5> > shp5 = evt.get(m_str_src, m_key_in, &m_src);
     	if (shp5.get()) { applyCorrections<T,TOUT>(evt, shp5->data()); return true; } 
      }

      if (m_ndim == 1) {
     	shared_ptr< ndarray<T,1> > shp1 = evt.get(m_str_src, m_key_in, &m_src);
     	if (shp1.get()) { applyCorrections<T,TOUT>(evt, shp1->data()); return true; } 
      }

      return false;
    }  

//-------------------

  template <typename T, typename TOUT>
    void applyCorrections(Event& evt, const T* p_rdata)
    {
     	  // 1) Evaluate: m_cdat[i] = (p_rdata[i] - m_peds[i] - m_norm*m_bkgd[i]) * m_gain[i]; 
     	  // 2) apply mask: m_mask[i];
     	  // 3) apply constant threshold: m_low_thre;
     	  // 4) apply nRMS threshold: m_low_nrms*m_nrms_data[i];

	  //m_cdat = new double[m_size];
     	  //memcpy(m_cdat,m_rdat,m_size*sizeof(double)); 
          //MsgLog( name(), info, "m_shape: " << m_shape[0] << " " << m_shape[1] );

	  m_count_get++;

	  //ndarray<TOUT,2> cdata(m_shape);
	  //TOUT* p_cdata = cdata.data();

	  TOUT low_val  = (TOUT) m_low_val; 
	  TOUT low_thre = (TOUT) m_low_thre; 
	  TOUT mask_val = (TOUT) m_mask_val; 

     	  for(unsigned i=0; i<m_size; i++) p_cdata[i] = (TOUT)p_rdata[i];

     	  if (m_do_peds) {                    for(unsigned i=0; i<m_size; i++) p_cdata[i] -= (TOUT) m_peds_data[i];         }
     	  if (m_do_bkgd) { normBkgd(p_cdata); for(unsigned i=0; i<m_size; i++) p_cdata[i] -= (TOUT)(m_bkgd_data[i]*m_norm); }
     	  if (m_do_gain) {                    for(unsigned i=0; i<m_size; i++) p_cdata[i] *= (TOUT) m_gain_data[i];         }

     	  if (m_do_mask) {             
     	    for(unsigned i=0; i<m_size; i++) {
     	      if (m_mask_data[i]!=0) p_cdata[i] = mask_val; 
     	    }
     	  }

     	  if (m_do_thre) {             
     	    for(unsigned i=0; i<m_size; i++) {
     	      if (p_cdata[i] < low_thre) p_cdata[i] = low_val; 
     	    }
     	  }

     	  if (m_do_nrms) {             
     	    for(unsigned i=0; i<m_size; i++) {
     	      if (p_cdata[i] < (TOUT) m_nrms_data[i]) p_cdata[i] = low_val; 
     	    }
     	  }

          // if( m_print_bits &  8 ) MsgLog( name(), info, stringOf2DArrayData<T>(*ndarr.get(), std::string("Raw ndarr data:")) );
          // if( m_print_bits & 16 ) MsgLog( name(), info, stringOf2DArrayData<TOUT>(cdata, std::string("Calibr. data:")) );
 	  
          saveNDArrInEvent <TOUT> (evt, m_src, m_key_out, p_cdata, m_ndarr_pars, 1);
    }  

//-------------------
//-------------------

  template <typename T>
    void normBkgd(const T* p_cdata)
    {
      double sum_data=0;
      double sum_bkgd=0;
      for(std::vector<unsigned>::const_iterator it = v_inds.begin(); it != v_inds.end(); ++ it) {
        sum_data += p_cdata[*it];
        sum_bkgd += m_bkgd_data[*it];
        //sum_bkgd += m_bkgd->data()[*it];
      }
      m_norm = (sum_bkgd != 0)? (float)(sum_data/sum_bkgd) : 1;
    }

//--------------------
//--------------------

  template <typename T>
  bool getConfigParsForType(Env& env)
  {
      shared_ptr<T> config = env.configStore().get(m_str_src, &m_src);
      if (config) {

        std::string str_src = srcToString(m_src); 

        WithMsgLog(name(), info, str) {
          str << "Get configuration parameters for source: " << str_src << "\n";
          //str << " roiMask = "          << config->roiMask();
          //str << " m_numAsicsStored = " << config->numAsicsStored();
         }  
        return true;
      }
      return false;
  }
//--------------------

};

} // namespace ImgAlgos

#endif // IMGALGOS_NDARRCALIB_H
