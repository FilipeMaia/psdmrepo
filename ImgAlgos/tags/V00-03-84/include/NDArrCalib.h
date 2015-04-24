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
#include "ImgAlgos/TimeInterval.h"

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

  typedef double data_proc_t;

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
  void checkOutTypeImplementation();
  void getCalibPars(Event& evt, Env& env);
  //void getConfigPars(Env& env);
  void defImgIndexesForBkgdNorm();
  void initAtFirstGetNdarray(Event& evt, Env& env);
  void procEvent(Event& evt, Env& env);
  void printCommonModePars();
  void printMaskAndBkgd();
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
  std::string     m_outtype;          // output ndarray data type; double (df), float, int int16
  unsigned        m_print_bits;       // bit mask for print options
  long            m_count_event;      // local event counter
  long            m_count_get;        // local successful get() counter
  long            m_count_msg;        // counts messages to constrain printout
  NDArrPars*      m_ndarr_pars;       // holds input data ndarray parameters
  unsigned        m_ndim;             // rank of the input data ndarray 
  unsigned        m_size;             // number of elements in the input data ndarray 
  DATA_TYPE       m_dtype;            // numerated data type for input array
  DATA_TYPE       m_ptype;            // numerated data type for processing array
  DATA_TYPE       m_otype;            // numerated data type for output array
  DETECTOR_TYPE   m_dettype;          // numerated detector type source

  const PSCalib::CalibPars::pedestals_t*     m_peds_data;
  const PSCalib::CalibPars::pixel_gain_t*    m_gain_data;
  const PSCalib::CalibPars::pixel_status_t*  m_stat_data;
  const PSCalib::CalibPars::common_mode_t*   m_cmod_data;
  const PSCalib::CalibPars::pixel_rms_t*     m_rms_data;

  bool                                 m_is_done_once;
  PSCalib::CalibPars::pixel_bkgd_t*    m_bkgd_data;
  PSCalib::CalibPars::pixel_mask_t*    m_mask_data;
  data_proc_t*                         m_nrms_data;

  PSCalib::CalibPars* m_calibpars;    // pointer to calibration store
  //data_proc_t* p_cdata;                // pointer to calibrated data array

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

      if ( m_print_bits & 128 ) MsgLog( name(), info, "mode:" << mode 
                                                 << "  dettype:" << m_dettype
                                                 << "  source:" << m_str_src);

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

      // Algorithm 1 for any detector 
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

      // Algorithm 2 - MEAN for any detector 
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

      // Algorithm 3 - MEDIAN for any detector 
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


      // Algorithm 4 - MEDIAN algorithm, detector-dependent
      else if ( mode == 4 ) { 

        TimeInterval dt;
	if (do_common_mode_median<T>(data)) {
          if ( m_print_bits & 128 ) MsgLog("medianInRegion", info, " common mode dt(sec) = " << dt.getCurrentTimeInterval() );
          return; 
	}

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

  template <typename T>
    bool do_common_mode_median(T* data)
    {
      const PSCalib::CalibPars::common_mode_t* pars = &m_cmod_data[1]; 
      // [0] element=mode is excluded from parameters
      unsigned cmtype = (unsigned) m_cmod_data[1];

      unsigned pbits = ( m_print_bits & 256 ) ? 0xffff : 0;

      // EPIX100A, common_mode file example: 4 1 20
      if ( m_dettype == EPIX100A ) {

        if ( m_print_bits & 128 ) MsgLog( name(), info, "EPIX100A cmtype:" << cmtype);

        //T maxCorrection = (T) m_cmod_data[2];

	unsigned shape[2] = {704, 768};

	size_t nregs  = 16;	
	size_t nrows  = shape[0]/2;
	size_t ncols  = shape[1]/8;
	size_t rowmin = 0;
        size_t colmin = 0;
        
        ndarray<T,2> d(data, shape);
        ndarray<const uint16_t,2> stat(m_stat_data, shape);
	
	for(size_t s=0; s<nregs; s++) {
	  //meanInRegion  <T>(pars, d, stat, rowmin[s], colmin[s], nrows, ncols, 1, 1); 
          rowmin = (s/8)*nrows;
          colmin = (s%8)*ncols;

  	  if ( cmtype & 1 ) {
            //common mode for 352x96-pixel 16 banks
	    medianInRegion<T>(pars, d, stat, rowmin, colmin, nrows, ncols, 1, 1, pbits); 
	  }

	  if ( cmtype & 2 ) {
            //common mode for 96-pixel rows in 16 banks
	    for(size_t r=0; r<nrows; r++) {
	      medianInRegion<T>(pars, d, stat, rowmin+r, colmin, 1, ncols, 1, 1, pbits); 
	    }
          }

	  if ( cmtype & 4 ) {
            //common mode for 352-pixel columns in 16 banks
	    for(size_t c=0; c<ncols; c++) {
	      medianInRegion<T>(pars, d, stat, rowmin, colmin+c, nrows, 1, 1, 1, pbits); 
	    }
          }
	}
       return true; 
      }

      // FCCD960, common_mode file example: 4 1 20
      else if ( m_dettype == FCCD960 ) {

        if ( m_print_bits & 128 ) MsgLog( name(), info, "FCCD960 cmtype:" << cmtype);

        //T maxCorrection = (T) m_cmod_data[2];

	unsigned shape[2] = {960, 960};

        ndarray<T,2> d(data, shape);
        ndarray<const uint16_t,2> stat(m_stat_data, shape);
	
  	if ( cmtype & 1 ) {
          //common mode correction for 1x160-pixel rows with stride 2
	  size_t nregs  = 6;	
	  size_t ncols  = shape[1]/nregs; // 160-pixel
	  size_t nrows  = 1;
          size_t colmin = 0;

	  for(size_t row=0; row<shape[1]; row++) {
	    for(size_t s=0; s<nregs; s++) {
	      colmin = s * ncols;
	      for(size_t k=0; k<2; k++)
	        medianInRegion<T>(pars, d, stat, row, colmin+k, nrows, ncols, 1, 2, pbits); 
	    }
	  }
	}

  	if ( cmtype & 2 ) {
          //common mode correction for 480x10-pixel 96*2 supercolumns
	  size_t nregs  = 96*2;	
	  size_t nrows  = shape[0]/2;
	  size_t ncols  = shape[1]/96;
        
	  for(size_t s=0; s<nregs; s++) {
	    //meanInRegion  <T>(pars, d, stat, rowmin[s], colmin[s], nrows, ncols, 1, 1); 
            size_t rowmin = (s/96)*nrows;
            size_t colmin = (s%96)*ncols;
	    medianInRegion<T>(pars, d, stat, rowmin, colmin, nrows, ncols, 1, 1, pbits); 
	  }
	}
        return true; 
      }

      return false; 
    }

//-------------------

  template <typename T, typename TPROC>
    // !!!!!!!!!!!!! void applyCorrections(Event& evt, const T* p_rdata, TPROC* p_cdata)
    void applyCorrections(Event& evt, T* p_rdata, TPROC* p_cdata)
    {
     	  // 1) Evaluate: m_cdat[i] = (p_rdata[i] - m_peds[i] - comm_mode[...] - norm*m_bkgd[i]) * m_gain[i]; 
     	  // 2) apply bad pixel status: m_stat[i];
     	  // 3) apply mask: m_mask[i];
     	  // 4) apply constant threshold: m_low_thre;
     	  // 5) apply nRMS threshold: m_low_nrms*m_nrms_data[i];

	  m_count_get++;

	  TPROC low_val  = (TPROC) m_low_val; 
	  TPROC low_thre = (TPROC) m_low_thre; 
	  TPROC mask_val = (TPROC) m_mask_val; 

     	  for(unsigned i=0; i<m_size; i++) p_cdata[i] = (TPROC)p_rdata[i];

     	  if (m_do_peds) { for(unsigned i=0; i<m_size; i++) p_cdata[i] -= (TPROC) m_peds_data[i]; }
     	  if (m_do_cmod && m_do_peds) { do_common_mode<TPROC>(p_cdata); }
     	  if (m_do_bkgd) { 
                           double norm = normBkgd(p_cdata);  
                           for(unsigned i=0; i<m_size; i++) p_cdata[i] -= (TPROC)(m_bkgd_data[i]*norm); 
          }
     	  if (m_do_gain) { for(unsigned i=0; i<m_size; i++) p_cdata[i] *= (TPROC) m_gain_data[i]; }

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
 	  
          //saveNDArrInEvent <TPROC> (evt, m_src, m_key_out, p_cdata, m_ndarr_pars, 1);
    }  

//-------------------

  template <typename TIN, typename TOUT, unsigned NDim>
  void saveNDArrayInEventChangeType(PSEvt::Event& evt, ndarray<TIN,NDim>& nda)
  {
    ndarray<TOUT,NDim> out_nda(nda.shape());

    // Pixel-by-pixel copy of input to output ndarray with type conversion:
    typename ndarray<TIN,NDim>::iterator it = nda.begin(); 
    typename ndarray<TOUT,NDim>::iterator it_out = out_nda.begin(); 
    for (; it!=nda.end(); ++it, ++it_out) {
        *it_out = (TOUT)*it;
    } 

    saveNDArrayInEvent<TOUT,NDim>(evt, m_src, m_key_out, out_nda); 
    return;
  }

//-------------------

  template <typename T, unsigned NDim>
  void saveNDArrayInEventForOutType(PSEvt::Event& evt, ndarray<T,NDim>& nda)
  {
    if(m_otype == m_ptype) {saveNDArrayInEvent<T,NDim>(evt, m_src, m_key_out, nda); return; }

    if(m_otype == FLOAT  ) {saveNDArrayInEventChangeType<T,float,NDim>   (evt, nda); return; }
    if(m_otype == INT    ) {saveNDArrayInEventChangeType<T,int,NDim>     (evt, nda); return; }
    if(m_otype == INT16  ) {saveNDArrayInEventChangeType<T,int16_t,NDim> (evt, nda); return; }
    if(m_otype == DOUBLE ) {saveNDArrayInEventChangeType<T,double,NDim>  (evt, nda); return; }
  }
 
//-------------------

  template <typename T, typename TPROC, unsigned NDim>
    bool procEventForTypeNDim(Event& evt)
    {
      if (m_ndim != NDim) return false;

      shared_ptr< ndarray<T,NDim> > shp = evt.get(m_str_src, m_key_in, &m_src);
      if (shp.get()) { 
        ndarray<TPROC,NDim> out_nda(shp->shape());
        applyCorrections<T,TPROC>(evt, shp->data(), out_nda.data()); 
        saveNDArrayInEventForOutType<TPROC,NDim>(evt, out_nda);
        return true;
      } 
      return false;
    }  

//-------------------

  template <typename T, typename TPROC>
    bool procEventForType(Event& evt)
    {
      // CONST
      if (m_ndim == 2 && procEventForTypeNDim<const T,TPROC,2>(evt)) return true;
      if (m_ndim == 3 && procEventForTypeNDim<const T,TPROC,3>(evt)) return true;
      if (m_ndim == 4 && procEventForTypeNDim<const T,TPROC,4>(evt)) return true;
      if (m_ndim == 5 && procEventForTypeNDim<const T,TPROC,5>(evt)) return true;
      if (m_ndim == 1 && procEventForTypeNDim<const T,TPROC,1>(evt)) return true;

      // NON-CONST
      if (m_ndim == 2 && procEventForTypeNDim<T,TPROC,2>(evt)) return true;
      if (m_ndim == 3 && procEventForTypeNDim<T,TPROC,3>(evt)) return true;
      if (m_ndim == 4 && procEventForTypeNDim<T,TPROC,4>(evt)) return true;
      if (m_ndim == 5 && procEventForTypeNDim<T,TPROC,5>(evt)) return true;
      if (m_ndim == 1 && procEventForTypeNDim<T,TPROC,1>(evt)) return true;
      return false;
    }  

//-------------------
};

} // namespace ImgAlgos

#endif // IMGALGOS_NDARRCALIB_H
