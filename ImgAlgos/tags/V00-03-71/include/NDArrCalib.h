#ifndef IMGALGOS_NDARRCALIB_H
#define IMGALGOS_NDARRCALIB_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class NDArrCalib.
//      Apply corrections to ndarray using pedestals, common mode, background, gain factor, and mask.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"
#include "MsgLogger/MsgLogger.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ImgAlgos/CommonMode.h"
#include "ImgAlgos/GlobalMethods.h"
#include "PSCalib/CalibPars.h"

//#include "ImgAlgos/ImgParametersV1.h"
//#include "PSCalib/PnccdCalibPars.h"
//#include "PSCalib/CSPad2x2CalibPars.h"
#include "psalg/psalg.h"

//#include "psddl_psana/pnccd.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgAlgos {

/// @addtogroup ImgAlgos

/**
 *  Apply corrections to ndarray using pedestals, common mode, background, gain factor, and mask.
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
  void getCalibPars(Event& evt, Env& env);
  //void getConfigPars(Env& env);
  void defImgIndexesForBkgdNorm();
  void initAtFirstGetNdarray(Event& evt, Env& env);
  void procEvent(Event& evt, Env& env);
  void printCommonModePars();
  void findDetectorType();

private:

  Pds::Src        m_src;              // source address of the data object
  Source          m_str_src;          // string with source name
  std::string     m_key_in;           // string with key for input data
  std::string     m_key_out;          // string with key for output image
  bool            m_do_peds;          // flag: true = do pedestal subtraction
  bool            m_do_cmod;          // flag: true = do common mode subtraction
  bool            m_do_stat;          // flag: true = mask hot/bad pixels from pixel_status file 
  bool            m_do_mask;          // flag: true = apply mask from file
  bool            m_do_bkgd;          // flag: true = subtract background
  bool            m_do_gain;          // flag: true = apply the gain correction
  bool            m_do_nrms;          // flag: true = apply the threshold as nRMS
  bool            m_do_thre;          // flag: true = apply the threshold
  std::string     m_fname_mask;       // string file name for mask
  std::string     m_fname_bkgd;       // string file name for background
  double          m_mask_val;         // Value substituted for masked bits
  double          m_low_nrms;         // The low threshold number of RMS
  double          m_low_thre;         // The low threshold
  double          m_low_val;          // The value of substituting amplitude below threshold
  unsigned        m_ind_min;          // window for background normalization from      m_ind_min
  unsigned        m_ind_max;          // window for background normalization to        m_ind_max
  unsigned        m_ind_inc;          // window for background normalization increment m_ind_inc
  unsigned        m_print_bits;       // bit mask for print options
  long            m_count_event;      // local event counter
  long            m_count_get;        // local successful get() counter
  long            m_count_msg;        // counts messages to constrain printout
  NDArrPars*      m_ndarr_pars;       // holds input data ndarray parameters
  unsigned        m_ndim;             // rank of the input data ndarray 
  unsigned        m_size;             // number of elements in the input data ndarray 
  DATA_TYPE       m_dtype;            // numerated data type for data array
  DETECTOR_TYPE   m_dettype;          // numerated detector type source

  const PSCalib::CalibPars::pedestals_t*     m_peds_data;
  const PSCalib::CalibPars::pixel_gain_t*    m_gain_data;
  const PSCalib::CalibPars::pixel_status_t*  m_stat_data;
  const PSCalib::CalibPars::common_mode_t*   m_cmod_data;
  const PSCalib::CalibPars::pixel_rms_t*     m_rms_data;

  PSCalib::CalibPars::pixel_bkgd_t*    m_bkgd_data;
  PSCalib::CalibPars::pixel_mask_t*    m_mask_data;
  data_out_t*                          m_nrms_data;

  PSCalib::CalibPars* m_calibpars;    // pointer to calibration store
  //data_out_t* p_cdata;                // pointer to calibrated data array

  std::vector<unsigned> v_inds;       // vector of the image indexes for background normalization

//-------------------
// Evaluate normalization factor for background subtraction

  template <typename T>
    double normBkgd(const T* p_cdata)
    {
      double sum_data=0;
      double sum_bkgd=0;
      for(std::vector<unsigned>::const_iterator it = v_inds.begin(); it != v_inds.end(); ++ it) {
        sum_data += p_cdata[*it];
        sum_bkgd += m_bkgd_data[*it];
        //sum_bkgd += m_bkgd->data()[*it];
      }
      double norm = (sum_bkgd != 0)? (double)(sum_data/sum_bkgd) : 1;
      return norm;
    }

//-------------------

  template <typename T>
    void do_common_mode(T* data)
    {
      unsigned mode = (unsigned) m_cmod_data[0];
      const PSCalib::CalibPars::common_mode_t* pars = &m_cmod_data[1]; // [0] element=mode is excluded from parameters
      float cmod_corr = 0;

      if ( m_print_bits & 128 ) MsgLog( name(), info, "mode:" << mode << "  dettype:" << m_dettype);

      if ( mode == 0 ) return;

      // Algorithm 1 for CSPAD
      if ( mode == 1 && m_dettype == CSPAD ) {
          unsigned ssize = 185*388;
	  for (unsigned ind = 0; ind<32*ssize; ind+=ssize) {
	    cmod_corr = findCommonMode<T>(pars, &data[ind], &m_stat_data[ind], ssize); 
	  }
          return;
      }

      // Algorithm 1 for CSPAD2X2
      else if ( mode == 1 && m_dettype == CSPAD2X2 ) {
	  unsigned ssize = 185*388;
	  int stride = 2;
	  for (unsigned seg = 0; seg<2; ++seg) {
	    cmod_corr = findCommonMode<T>(pars, &data[seg], &m_stat_data[seg], ssize, stride); 
	  }
          return;
      }

      // Algorithm 1 for other detectors 
      else if ( mode == 1 ) {  
	//unsigned mode     = m_cmod_data[0]; // mode - algorithm number for common mode
	//unsigned mean_max = m_cmod_data[1]; // maximal value for the common mode correctiom
	//unsigned rms_max  = m_cmod_data[2]; // maximal value for the found peak rms
	//unsigned thresh   = m_cmod_data[3]; // threshold on number of pixels in the peak finding algorithm
	  unsigned nsegs    = (unsigned)m_cmod_data[4]; // number of segments in the detector
	  unsigned ssize    = (unsigned)m_cmod_data[5]; // segment size
	  unsigned stride   = (unsigned)m_cmod_data[6]; // stride (step to jump)

          nsegs  = (nsegs<1)   ?   1 : nsegs;
          ssize  = (ssize<100) ? 128 : ssize;
          stride = (nsegs<1)   ?   1 : stride;

	  for (unsigned ind = 0; ind<nsegs*ssize; ind+=ssize) {
	    cmod_corr = findCommonMode<T>(pars, &data[ind], &m_stat_data[ind], ssize, stride); 
	  }
          return;
      }

      // Algorithm 2 - MEAN
      else if (mode == 2) {
          T threshold     = (T)        m_cmod_data[1];
          T maxCorrection = (T)        m_cmod_data[2];
          unsigned length = (unsigned) m_cmod_data[3];
          T cm            = 0;          
          length = (length<100) ? 128 : length;
          
          for (unsigned i0=0; i0<m_size; i0+=length) {
	      psalg::commonMode<T>(&data[i0], &m_stat_data[i0], length, threshold, maxCorrection, cm);
              //     commonMode<T>(&data[i0], &m_stat_data[i0], length, threshold, maxCorrection, cm); // from GlobalMethods.h
          }
          return; 
      }

      // Algorithm 3 - MEDIAN
      else if (mode == 3) {
          T threshold     = (T)        m_cmod_data[1];
          T maxCorrection = (T)        m_cmod_data[2];
          unsigned length = (unsigned) m_cmod_data[3];
          T cm            = 0;          
          length = (length<100) ? 128 : length;
          
          for (unsigned i0=0; i0<m_size; i0+=length) {
              psalg::commonModeMedian<T>(&data[i0], &m_stat_data[i0], length, threshold, maxCorrection, cm);
          }
          return; 
      }

      // Algorithm 4 for EPIX100A    common_mode file example: 4 5 10
      else if ( mode == 4 && m_dettype == EPIX100A ) {

        //T threshold     = (T) m_cmod_data[1];
        //T maxCorrection = (T) m_cmod_data[2];

	unsigned shape[2] = {704, 768};

	//size_t nregs = 4;	
	//size_t nrows = shape[0]/2;
	//size_t ncols = shape[1]/2;
	//size_t rowmin[nregs]  = {0,     0, nrows, nrows};
	//size_t colmin[nregs]  = {0, ncols,     0, ncols};

	size_t nregs  = 16;	
	size_t nrows  = shape[0]/2;
	size_t ncols  = shape[1]/8;
	size_t rowmin = 0;
        size_t colmin = 0;
        
        ndarray<T,2> d(data, shape);
        ndarray<const uint16_t,2> stat(m_stat_data, shape);
	
        unsigned pbits = ( m_print_bits & 128 ) ? 0xffff : 0;

	for(size_t s=0; s<nregs; s++) {
	  //meanInRegion<T>(pars, d, stat, rowmin[s], colmin[s], nrows, ncols, 1, 1); 
	  //medianInRegion<T>(pars, d, stat, rowmin[s], colmin[s], nrows, ncols, 1, 1); 
	  
	  rowmin = (s/8)*nrows;
	  colmin = (s%8)*ncols;
	  
	  medianInRegion<T>(pars, d, stat, rowmin, colmin, nrows, ncols, 1, 1, pbits); 
        }
        return; 
      }

      // Other algorithms which are not implemented yet
      else {
	  static long counter = 0; counter ++;
	  if (counter<21) {  MsgLog( name(), warning, "Algorithm " << mode << " requested in common_mode parameters is not implemented.");
	    if (counter==20) MsgLog( name(), warning, "STOP PRINTING ABOVE WARNING MESSAGE.");
	  }
      }
    }

//-------------------

  template <typename T, typename TOUT>
    void applyCorrections(Event& evt, const T* p_rdata, TOUT* p_cdata)
    {
     	  // 1) Evaluate: m_cdat[i] = (p_rdata[i] - m_peds[i] - comm_mode[...] - norm*m_bkgd[i]) * m_gain[i]; 
     	  // 2) apply bad pixel status: m_stat[i];
     	  // 3) apply mask: m_mask[i];
     	  // 4) apply constant threshold: m_low_thre;
     	  // 5) apply nRMS threshold: m_low_nrms*m_nrms_data[i];

	  m_count_get++;

	  TOUT low_val  = (TOUT) m_low_val; 
	  TOUT low_thre = (TOUT) m_low_thre; 
	  TOUT mask_val = (TOUT) m_mask_val; 

     	  for(unsigned i=0; i<m_size; i++) p_cdata[i] = (TOUT)p_rdata[i];

     	  if (m_do_peds) { for(unsigned i=0; i<m_size; i++) p_cdata[i] -= (TOUT) m_peds_data[i]; }
     	  if (m_do_cmod && m_do_peds) { do_common_mode<TOUT>(p_cdata); }
     	  if (m_do_bkgd) { 
                           double norm = normBkgd(p_cdata);  
                           for(unsigned i=0; i<m_size; i++) p_cdata[i] -= (TOUT)(m_bkgd_data[i]*norm); 
          }
     	  if (m_do_gain) { for(unsigned i=0; i<m_size; i++) p_cdata[i] *= (TOUT) m_gain_data[i]; }

     	  if (m_do_stat) {             
     	    for(unsigned i=0; i<m_size; i++) {
     	      if (m_stat_data[i]!=0) p_cdata[i] = mask_val; // m_stat_data[i] == 0 - good pixel
     	    }
     	  }

     	  if (m_do_mask) {             
     	    for(unsigned i=0; i<m_size; i++) {
     	      if (m_mask_data[i]==0) p_cdata[i] = mask_val; // m_mask_data[i] == 1 - good pixel 
     	    }
     	  }

     	  if (m_do_thre) {             
     	    for(unsigned i=0; i<m_size; i++) {
     	      if (p_cdata[i] < low_thre) p_cdata[i] = low_val; 
     	    }
     	  }

     	  if (m_do_nrms) {             
     	    for(unsigned i=0; i<m_size; i++) {
     	      if (p_cdata[i] < m_nrms_data[i]) p_cdata[i] = low_val; 
     	    }
     	  }

          if( m_print_bits & 32 ) {
	    std::stringstream ss; ss<<"Raw data: "; for (int i=0; i<10; ++i) ss << " " << p_rdata[i];
            MsgLog( name(), info, ss.str());
	  }

          if( m_print_bits & 64 ) {
	    std::stringstream ss; ss<<"Corr data:"; for (int i=0; i<10; ++i) ss << " " << p_cdata[i];
            MsgLog( name(), info, ss.str());
          }
 	  
          //saveNDArrInEvent <TOUT> (evt, m_src, m_key_out, p_cdata, m_ndarr_pars, 1);
    }  
//-------------------

  template <typename T, typename TOUT>
    bool procEventForType(Event& evt)
    {
      // CONST

      if (m_ndim == 2) {
     	shared_ptr< ndarray<const T,2> > shp2_const= evt.get(m_str_src, m_key_in, &m_src);
     	if (shp2_const.get()) {
	  ndarray<TOUT,2> out_nda(shp2_const->shape());
          applyCorrections<T,TOUT>(evt, shp2_const->data(), out_nda.data()); 
          save2DArrayInEvent<TOUT>(evt, m_src, m_key_out, out_nda);
          return true;
        } 
      }

      if (m_ndim == 3) {
     	shared_ptr< ndarray<const T,3> > shp3_const = evt.get(m_str_src, m_key_in, &m_src);
     	if (shp3_const.get()) { 
	  ndarray<TOUT,3> out_nda(shp3_const->shape());
          applyCorrections<T,TOUT>(evt, shp3_const->data(), out_nda.data()); 
          save3DArrayInEvent<TOUT>(evt, m_src, m_key_out, out_nda);
          return true;
        } 
      }

      if (m_ndim == 4) {
     	shared_ptr< ndarray<const T,4> > shp4_const = evt.get(m_str_src, m_key_in, &m_src);
     	if (shp4_const.get()) { 
	  ndarray<TOUT,4> out_nda(shp4_const->shape());
          applyCorrections<T,TOUT>(evt, shp4_const->data(), out_nda.data());
          save4DArrayInEvent<TOUT>(evt, m_src, m_key_out, out_nda);
          return true;
        } 
      }

      if (m_ndim == 5) {
     	shared_ptr< ndarray<const T,5> > shp5_const = evt.get(m_str_src, m_key_in, &m_src);
     	if (shp5_const.get()) {
	  ndarray<TOUT,5> out_nda(shp5_const->shape());
          applyCorrections<T,TOUT>(evt, shp5_const->data(), out_nda.data());
          save5DArrayInEvent<TOUT>(evt, m_src, m_key_out, out_nda);
          return true;
        } 
      }

      if (m_ndim == 1) {
     	shared_ptr< ndarray<const T,1> > shp1_const = evt.get(m_str_src, m_key_in, &m_src);
     	if (shp1_const.get()) {
	  ndarray<TOUT,1> out_nda(shp1_const->shape());
          applyCorrections<T,TOUT>(evt, shp1_const->data(), out_nda.data());
          save1DArrayInEvent<TOUT>(evt, m_src, m_key_out, out_nda);
          return true;
        } 
      }

      // NON-CONST

      if (m_ndim == 2) {
     	shared_ptr< ndarray<T,2> > shp2 = evt.get(m_str_src, m_key_in, &m_src);
     	if (shp2.get()) { 
	  ndarray<TOUT,2> out_nda(shp2->shape());
          applyCorrections<T,TOUT>(evt, shp2->data(), out_nda.data()); 
          save2DArrayInEvent<TOUT>(evt, m_src, m_key_out, out_nda);
          return true;
        } 
      }

      if (m_ndim == 3) {
     	shared_ptr< ndarray<T,3> > shp3 = evt.get(m_str_src, m_key_in, &m_src);
     	if (shp3.get()) { 
	  ndarray<TOUT,3> out_nda(shp3->shape());
          applyCorrections<T,TOUT>(evt, shp3->data(), out_nda.data());
          save3DArrayInEvent<TOUT>(evt, m_src, m_key_out, out_nda);
          return true;
        } 
      }

      if (m_ndim == 4) {
     	shared_ptr< ndarray<T,4> > shp4 = evt.get(m_str_src, m_key_in, &m_src);
     	if (shp4.get()) { 
	  ndarray<TOUT,4> out_nda(shp4->shape());
          applyCorrections<T,TOUT>(evt, shp4->data(), out_nda.data());
          save4DArrayInEvent<TOUT>(evt, m_src, m_key_out, out_nda);
          return true;
        } 
      }

      if (m_ndim == 5) {
     	shared_ptr< ndarray<T,5> > shp5 = evt.get(m_str_src, m_key_in, &m_src);
     	if (shp5.get()) { 
	  ndarray<TOUT,5> out_nda(shp5->shape());
          applyCorrections<T,TOUT>(evt, shp5->data(), out_nda.data());
          save5DArrayInEvent<TOUT>(evt, m_src, m_key_out, out_nda);
          return true;
        } 
      }

      if (m_ndim == 1) {
     	shared_ptr< ndarray<T,1> > shp1 = evt.get(m_str_src, m_key_in, &m_src);
     	if (shp1.get()) { 
	  ndarray<TOUT,1> out_nda(shp1->shape());
          applyCorrections<T,TOUT>(evt, shp1->data(), out_nda.data());
          save1DArrayInEvent<TOUT>(evt, m_src, m_key_out, out_nda);
          return true;
        } 
      }

      return false;
    }  

//--------------------

/*
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
*/

//--------------------
//--------------------
};

} // namespace ImgAlgos

#endif // IMGALGOS_NDARRCALIB_H
