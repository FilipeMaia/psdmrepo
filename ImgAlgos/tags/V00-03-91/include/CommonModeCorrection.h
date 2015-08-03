#ifndef IMGALGOS_COMMONMODECORRECTION_H
#define IMGALGOS_COMMONMODECORRECTION_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description: see documentation below
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include <string>
//#include <iostream> // for cout
//#include <cstddef>  // for size_t
//#include <cstring>  // for memcpy

using namespace std;

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "PSEvt/Source.h"
#include "MsgLogger/MsgLogger.h"
#include "ndarray/ndarray.h"

#include "PSCalib/CalibPars.h"
#include "ImgAlgos/CommonMode.h"
#include "ImgAlgos/GlobalMethods.h"
#include "ImgAlgos/TimeInterval.h"
#include "psalg/psalg.h"

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
 *  @brief CommonModeCorrection - apply common mode correction algorithms.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 *
 *  @see CommonMode
 *
 *
 *  @anchor interface
 *  @par<interface> Interface Description
 *
 * 
 *  @li Includes
 *  @code
 *  #include "ImgAlgos/CommonModeCorrection.h"
 *  #include "ndarray/ndarray.h"     // need it for I/O arrays
 *  @endcode
 *
 *
 *  @li Initialization
 *  \n
 *  @code
 *  const PSEvt::Source source("DetInfo(CxiDs2.0:Cspad.0)"); 
 *  const common_mode_t cmod_pars[] = {1, 50, 25, 100};
 *  const unsigned size = 32*185*388;
 *  const pixel_status_t* status = 0; // or get it from calibration store
 *  const unsigned pbits = 0377;
 * 
 *  CommonModeCorrection* p_cmog = new CommonModeCorrection(source, cmod_pars, size status, pbits);
 *  @endcode
 *
 *
 *  @li Do common mode correction
 *  @code
 *  T* data = ...; // get it from data (assumes that pedestals are subtracted).
 *  p_cmog -> do_common_mode<T>(data);
 *  @endcode
 */

//template <typename T>
class CommonModeCorrection  {
public:

  typedef PSCalib::CalibPars::pixel_status_t pixel_status_t;
  typedef PSCalib::CalibPars::common_mode_t common_mode_t;

  //static const size_t RSMMAX = 10;

  /**
   * @brief Evaluate and apply common mode correction. 
   * 
   * @param[in] source    - data source
   * @param[in] cmod_pars - common mode parameters from calibration file
   * @param[in] size      - ndarray full saze for data and pixel-status array
   * @param[in] status    - pixel status array, 0-good/ >0-bad pixel
   * @param[in] pbits     - print control bit-word; =0-print nothing, +1-input parameters, ....
   */

  CommonModeCorrection (const PSEvt::Source& source, 
                        const common_mode_t* cmod_pars, 
                        const unsigned size, 
                        const pixel_status_t* status=0, 
                        const unsigned pbits=0);

  /// Destructor
  virtual ~CommonModeCorrection () {}

  /**
   * @brief Evaluate and apply common mode correction
   * 
   * ...
   */

  void printInputPars();

  // Copy constructor and assignment are disabled by default
  CommonModeCorrection ( const CommonModeCorrection& ) ;
  CommonModeCorrection& operator = ( const CommonModeCorrection& ) ;


private:

  PSEvt::Source          m_source;      // string with source name
  const common_mode_t*   m_cmod_pars;   // common mode parameters
  unsigned               m_size;        // number of elements in the input data ndarray 
  const pixel_status_t*  m_status;      // pointer to pixel status array
  unsigned               m_pbits;       // bit mask for print options
  DETECTOR_TYPE          m_dettype;     // numerated detector type source

  /// Returns name of the class
  inline const char* _name_() {return "ImgAlgos::CommonModeCorrection";}

public:
//-------------------

  template <typename T>
    void do_common_mode(T* data)
    {
      unsigned mode = (unsigned) m_cmod_pars[0];
      const common_mode_t* pars = &m_cmod_pars[1]; // [0] element=mode is excluded from parameters
      float cmod_corr = 0;

      const pixel_status_t* status = 0;

      if (m_pbits & 128) MsgLog(_name_(), info,   "mode:" << mode 
                                          << "  dettype:" << m_dettype
                                          <<  "  source:" << m_source);

      if (mode == 0) return;

      // Algorithm 1 for CSPAD
      if (mode == 1 && m_dettype == CSPAD) {
          unsigned ssize = 185*388;
	  for (unsigned ind = 0; ind<32*ssize; ind+=ssize) {
            if(m_status) status = &m_status[ind];
	    cmod_corr = findCommonMode<T>(pars, &data[ind], status, ssize); 
	  }
          return;
      }

      // Algorithm 1 for CSPAD2X2
      else if (mode == 1 && m_dettype == CSPAD2X2) {
	  unsigned ssize = 185*388;
	  int stride = 2;
	  for (unsigned seg = 0; seg<2; ++seg) {
            if(m_status) status = &m_status[seg];
	    cmod_corr = findCommonMode<T>(pars, &data[seg], status, ssize, stride); 
	  }
          return;
      }

      // Algorithm 1 for any detector 
      else if (mode == 1) {  
	//unsigned mode     = m_cmod_pars[0]; // mode - algorithm number for common mode
	//unsigned mean_max = m_cmod_pars[1]; // maximal value for the common mode correctiom
	//unsigned rms_max  = m_cmod_pars[2]; // maximal value for the found peak rms
	//unsigned thresh   = m_cmod_pars[3]; // threshold on number of pixels in the peak finding algorithm
	  unsigned nsegs    = (unsigned)m_cmod_pars[4]; // number of segments in the detector
	  unsigned ssize    = (unsigned)m_cmod_pars[5]; // segment size
	  unsigned stride   = (unsigned)m_cmod_pars[6]; // stride (step to jump)

          nsegs  = (nsegs<1)   ?   1 : nsegs;
          ssize  = (ssize<100) ? 128 : ssize;
          stride = (nsegs<1)   ?   1 : stride;

	  for (unsigned ind = 0; ind<nsegs*ssize; ind+=ssize) {
            if(m_status) status = &m_status[ind];
	    cmod_corr = findCommonMode<T>(pars, &data[ind], status, ssize, stride); 
	  }
          return;
      }

      // Algorithm 2 - MEAN for any detector 
      else if (mode == 2) {
          T threshold     = (T)        m_cmod_pars[1];
          T maxCorrection = (T)        m_cmod_pars[2];
          unsigned length = (unsigned) m_cmod_pars[3];
          T cm            = 0;          
          length = (length<100) ? 128 : length;
          
          for (unsigned i0=0; i0<m_size; i0+=length) {
              if(m_status) status = &m_status[i0];
	      psalg::commonMode<T>(&data[i0], status, length, threshold, maxCorrection, cm);
              //     commonMode<T>(&data[i0], status, length, threshold, maxCorrection, cm); // from GlobalMethods.h
          }
          return; 
      }

      // Algorithm 3 - MEDIAN for any detector 
      else if (mode == 3) {
          T threshold     = (T)        m_cmod_pars[1];
          T maxCorrection = (T)        m_cmod_pars[2];
          unsigned length = (unsigned) m_cmod_pars[3];
          T cm            = 0;          
          length = (length<100) ? 128 : length;
          
          for (unsigned i0=0; i0<m_size; i0+=length) {
              if(m_status) status = &m_status[i0];
              psalg::commonModeMedian<T>(&data[i0], status, length, threshold, maxCorrection, cm);
          }
          return; 
      }


      // Algorithm 4 - MEDIAN algorithm, detector-dependent
      else if (mode == 4) { 

        TimeInterval dt;
	if (do_common_mode_median<T>(data)) {
          if (m_pbits & 128) MsgLog("medianInRegion", info, " common mode dt(sec) = " << dt.getCurrentTimeInterval() );
          return; 
	}

      }


      // Other algorithms which are not implemented yet
      else {
	  static long counter = 0; counter ++;
	  if (counter<21) {  MsgLog(_name_(), warning, "Algorithm " << mode << " requested in common_mode parameters is not implemented.");
	    if (counter==20) MsgLog(_name_(), warning, "STOP PRINTING ABOVE WARNING MESSAGE.");
	  }
      }
    }

//-------------------

  template <typename T>
    bool do_common_mode_median(T* data)
    {
      const common_mode_t* pars = &m_cmod_pars[1]; 
      // [0] element=mode is excluded from parameters
      unsigned cmtype = (unsigned) m_cmod_pars[1];

      unsigned pbits = (m_pbits & 256) ? 0xffff : 0;

      // EPIX100A, common_mode file example: 4 1 20
      if (m_dettype == EPIX100A) {

        if ( m_pbits & 128 ) MsgLog(_name_(), info, "EPIX100A cmtype:" << cmtype);

        //T maxCorrection = (T) m_cmod_pars[2];

	unsigned shape[2] = {704, 768};

	size_t nregs  = 16;	
	size_t nrows  = shape[0]/2;
	size_t ncols  = shape[1]/8;
	size_t rowmin = 0;
        size_t colmin = 0;
        
        ndarray<T,2> d(data, shape);
        ndarray<const pixel_status_t,2> stat(m_status, shape);
	
	for(size_t s=0; s<nregs; s++) {
	  //meanInRegion  <T>(pars, d, stat, rowmin[s], colmin[s], nrows, ncols, 1, 1); 
          rowmin = (s/8)*nrows;
          colmin = (s%8)*ncols;

  	  if (cmtype & 1) {
            //common mode for 352x96-pixel 16 banks
	    medianInRegion<T>(pars, d, stat, rowmin, colmin, nrows, ncols, 1, 1, pbits); 
	  }

	  if (cmtype & 2) {
            //common mode for 96-pixel rows in 16 banks
	    for(size_t r=0; r<nrows; r++) {
	      medianInRegion<T>(pars, d, stat, rowmin+r, colmin, 1, ncols, 1, 1, pbits); 
	    }
          }

	  if (cmtype & 4) {
            //common mode for 352-pixel columns in 16 banks
	    for(size_t c=0; c<ncols; c++) {
	      medianInRegion<T>(pars, d, stat, rowmin, colmin+c, nrows, 1, 1, 1, pbits); 
	    }
          }
	}
        return true; 
      }

      // FCCD960, common_mode file example: 4 1 20
      else if (m_dettype == FCCD960) {

        if (m_pbits & 128) MsgLog(_name_(), info, "FCCD960 cmtype:" << cmtype);

        //T maxCorrection = (T) m_cmod_pars[2];

	unsigned shape[2] = {960, 960};

        ndarray<T,2> d(data, shape);
        ndarray<const pixel_status_t,2> stat(m_status, shape);
	
  	if (cmtype & 1) {
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

  	if (cmtype & 2) {
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
};

} // namespace ImgAlgos

#endif // IMGALGOS_COMMONMODECORRECTION_H
