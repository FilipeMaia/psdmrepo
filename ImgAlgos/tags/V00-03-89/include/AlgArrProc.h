#ifndef IMGALGOS_ALGARRPROC_H
#define IMGALGOS_ALGARRPROC_H

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
#include <vector>
#include <iostream> // for cout
#include <stdint.h> // uint8_t, uint32_t, etc.
#include <cstddef>  // for size_t
#include <cstring>  // for memcpy
//#include <math.h>   // for exp
#include <cmath>    // for sqrt
#include <stdexcept>
#include <algorithm> // for fill_n

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSCalib/CalibPars.h"  // for pixel_mask_t
#include "MsgLogger/MsgLogger.h"
#include "ndarray/ndarray.h"
#include "ImgAlgos/GlobalMethods.h"
#include "ImgAlgos/Window.h"

#include "ImgAlgos/AlgImgProc.h"


using namespace std;


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
 *  @brief AlgArrProc - is a smearing algorithm for ndarray<T,2> using 2-d Gaussian weights.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 *
 *  @see AlgArrProc
 *
 *
 *  @anchor interface
 *  @par<interface> Interface Description
 *
 *
 * 
 *  @li  Include
 *  @code
 *  #include "ImgAlgos/AlgArrProc.h"
 *  #include "ndarray/ndarray.h"     // need it for I/O arrays
 *  @endcode
 *
 *
 *
 *  @li Initialization
 *  \n
 *  @code
 *  size_t      seg    = 2;
 *  size_t      rowmin = 10;
 *  size_t      rowmax = 170;
 *  size_t      colmin = 100;
 *  size_t      colmax = 200;
 *  unsigned    pbits  = 0177777;
 * 
 *  ImgAlgos::AlgArrProc* algson = new ImgAlgos::AlgArrProc(seg, rowmin, rowmax, colmin, colmax, pbits);
 *  @endcode
 *
 *  @li Set parameters for S/N
 *  @code
 *  float       r0 = 5;
 *  float       dr = 0.05;
 *  aip->setSoNPars(r0,dr);
 *  @endcode
 *
 *  @li Get S/N
 *  @code
 *  ndarray<const T,2> data = ....;    // input raw ndarray
 *  ndarray<mask_t,2>  mask = ....;    // input mask ndarray, may be omitted
 *  ndarray<mask_t,2>  son;            // output S/N ndarray
 *
 *  algson->SoN<T>(data, mask, son);
 *  @endcode
 *
 *  @li Print methods
 *  @code
 *  algson->printInputPars();
 *  algson->printIndexes();
 *  @endcode
 */

/*
struct SegWindow {
  size_t segind;
  Window window;
};
*/

class AlgArrProc {
public:

  typedef PSCalib::CalibPars::pixel_mask_t mask_t;
  typedef uint32_t wind_t;
  //typedef ImgAlgos::Peak Peak;

  /*
  typedef unsigned shape_t;
  typedef uint32_t conmap_t;
  typedef uint16_t pixel_status_t;
  typedef float son_t;
  */

  /**
   * @brief Class constructor is used for initialization of all paramaters. 
   * 
   * @param[in] nda_winds - ndarray with windows
   * @param[in] pbits     - print control bit-word; =0-print nothing, +1-input parameters, +2-algorithm area, etc.
   */

  AlgArrProc();
  AlgArrProc(const unsigned& pbits);
  AlgArrProc(ndarray<const wind_t,2> winds, const unsigned& pbits=0);
  //AlgArrProc(ndarray<const wind_t,2>& nda_winds, const unsigned& pbits=0);
  //AlgArrProc(ndarray<const wind_t,2>* nda_winds=0, const unsigned& pbits=0);
  //AlgArrProc(ndarray<const wind_t,2>& nda_winds, const unsigned& pbits=0) { AlgArrProc(&nda_winds, pbits); };


  //AlgArrProc(std::vector<Window>& vec_winds, const unsigned& pbits=0);

  void setWindows(ndarray<const wind_t,2> nda_winds);

  /// Set peak selection parameters MUST BE CALLED BEFORE PEAKFINDER!!!
  void setPeakSelectionPars(const float& npix_min=2, 
                            const float& npix_max=200, 
                            const float& amax_thr=0, 
                            const float& atot_thr=0,
                            const float& son_min=3);

  //void setSoNPars(const float& r0=5, const float& dr=0.05) { std::cout << "AlgArrProc::setSoNPars IS NOT IMPLEMENTED!!!"; }

  /// Destructor
  virtual ~AlgArrProc (); 

  /// Prints memeber data
  void printInputPars();

private:

  unsigned   m_pbits;            // pirnt control bit-word

  DATA_TYPE  m_dtype;            // numerated data type for data array
  unsigned   m_ndim;             // ndarray number of dimensions
  size_t     m_nsegs;
  size_t     m_nrows;
  size_t     m_ncols;
  size_t     m_stride;
  unsigned   m_sshape[2];
  bool       m_is_inited;

  mask_t*    m_mask_def;
  const mask_t* m_mask;

  float    m_peak_npix_min; // peak selection parameter
  float    m_peak_npix_max; // peak selection parameter
  float    m_peak_amax_thr; // peak selection parameter
  float    m_peak_atot_thr; // peak selection parameter
  float    m_peak_son_min;  // peak selection parameter

  std::vector<Window>        v_winds;
  std::vector<AlgImgProc*>   v_algip;    // vector of pointers to the AlgImgProc objects for windows

  /// Returns string name of the class for messanger
  inline const char* _name() {return "ImgAlgos::AlgArrProc";}

  /// Returns ndarray of peaks evaluated by any of peakfinders 
  const ndarray<const Peak, 1>  _ndarrayOfPeaks(const unsigned& npeaks);

  /// Returns ndarray of peak-float pars evaluated by any of peakfinders 
  const ndarray<const float, 2> _ndarrayOfPeakPars(const unsigned& npeaks);

  //AlgArrProc ( const AlgArrProc& ) ;
  //AlgArrProc& operator = ( const AlgArrProc& ) ;


//--------------------

public:

//--------------------
//--------------------

  template <typename T, unsigned NDim>
  bool
  _initAlgImgProc(const ndarray<const T, NDim>& data, const ndarray<const mask_t, NDim>& mask)
  {
    if(m_pbits & 256) MsgLog(_name(), info, "in _initAlgImgProc");

    m_mask = (mask.size()) ? mask.data() : m_mask_def;

    if(m_is_inited) return true;

    if(data.empty()) return false;

    m_ndim   = NDim;
    if(m_ndim < 2) throw std::runtime_error("Non-acceptable number of dimensions < 2 in input ndarray");

    m_dtype  = dataType<T>();
    m_ncols  = data.shape()[m_ndim-1];
    m_nrows  = data.shape()[m_ndim-2];
    m_nsegs  = (m_ndim>2) ? data.size()/m_ncols/m_nrows : 1;
    m_stride = m_ncols*m_nrows;

    m_sshape[0] = m_nrows;
    m_sshape[1] = m_ncols;

    m_is_inited = true;

    unsigned pbits = m_pbits; // 0177777;

    if(v_winds.empty()) {
        // ALL segments will be processed
        if(m_pbits & 256) MsgLog(_name(), info, "List of windows is empty, all sensors will be processed.")
        v_algip.reserve(m_nsegs);
      
        for(size_t seg=0; seg<m_nsegs; ++seg) {            
            AlgImgProc* p_alg = new AlgImgProc(seg, 0, m_nrows, 0, m_ncols, pbits);
            v_algip.push_back(p_alg);
            p_alg->setPeakSelectionPars(m_peak_npix_min, m_peak_npix_max, m_peak_amax_thr, m_peak_atot_thr, m_peak_son_min);
        }       
    }
    else {
        // Windows ONLY will be processed
        if(m_pbits & 256) MsgLog(_name(), info, "Windows from the list will be processed.")
        for(std::vector<Window>::iterator it = v_winds.begin(); it != v_winds.end(); ++ it) {
            AlgImgProc* p_alg = new AlgImgProc(it->segind, it->rowmin, it->rowmax, it->colmin, it->colmax , pbits);
            v_algip.push_back(p_alg); 
            p_alg->setPeakSelectionPars(m_peak_npix_min, m_peak_npix_max, m_peak_amax_thr, m_peak_atot_thr, m_peak_son_min);
        }
    }

    //if(mask.empty()) {
    if(mask.size() == 0) {
        // Define default mask
        if (! m_mask_def) delete m_mask_def;    
        m_mask_def = new mask_t[data.size()];
        m_mask = m_mask_def;
        std::fill_n(m_mask_def, int(data.size()), mask_t(1));
        if(m_pbits & 256) MsgLog(_name(), info, "Mask is empty, all pixels will be processed.")
    } else {
        if(m_pbits & 256) MsgLog(_name(), info, "Mask is used for pixel processing.")
    }

    return true;
  }

//--------------------
//--------------------

  template <typename T, unsigned NDim>
  unsigned
  numberOfPixAboveThr( const ndarray<const T, NDim> data
		     , const ndarray<const mask_t, NDim> mask
                     , const T& thr
                     )
  {
    if(m_pbits & 256) MsgLog(_name(), info, "in numberOfPixAboveThr " << thr);

    if(! _initAlgImgProc<T,NDim>(data, mask)) return 0;

    unsigned counter = 0;

    for (std::vector<AlgImgProc*>::iterator it = v_algip.begin(); it != v_algip.end(); ++it) {

        size_t ind = (*it)->segind() * m_stride;              
	const ndarray<const T,2>      seg_data(&data.data()[ind], m_sshape);
	const ndarray<const mask_t,2> seg_mask(&m_mask[ind], m_sshape);

        counter += (*it) -> numberOfPixAboveThr<T>(seg_data, seg_mask, thr);
    }
    return counter;
  }

//--------------------
//--------------------

  template <typename T, unsigned NDim>
  double
  intensityOfPixAboveThr( const ndarray<const T, NDim> data
                        , const ndarray<const mask_t, NDim> mask
                        , const T& thr
                        )
  {
    if(m_pbits & 256) MsgLog(_name(), info, "in intensityOfPixAboveThr" << thr);

    if(! _initAlgImgProc<T,NDim>(data, mask)) return 0;

    double intensity = 0;

    for (std::vector<AlgImgProc*>::iterator it = v_algip.begin(); it != v_algip.end(); ++it) {

        size_t ind = (*it)->segind() * m_stride;              
	const ndarray<const T,2>      seg_data(&data.data()[ind], m_sshape);
	const ndarray<const mask_t,2> seg_mask(&m_mask[ind], m_sshape);

        intensity += (*it) -> intensityOfPixAboveThr<T>(seg_data, seg_mask, thr);
    }
    return intensity;
  }

//--------------------
//--------------------

  template <typename T, unsigned NDim>
  ndarray<const float, 2>
  dropletFinder( const ndarray<const T, NDim> data
               , const ndarray<const mask_t, NDim> mask
               , const T& thr_low
               , const T& thr_high
               , const unsigned& rad=5
               , const float& dr=0.05
               )
  {
    if(m_pbits & 256) MsgLog(_name(), info, "in dropletFinder");

    if(! _initAlgImgProc<T,NDim>(data, mask)) { ndarray<const float, 2> empty; return empty; }

    unsigned npeaks = 0;

    for (std::vector<AlgImgProc*>::iterator it = v_algip.begin(); it != v_algip.end(); ++it) {

        size_t ind = (*it)->segind() * m_stride;              
	const ndarray<const T,2>      seg_data(&data.data()[ind], m_sshape);
	const ndarray<const mask_t,2> seg_mask(&m_mask[ind], m_sshape);

        std::vector<Peak>& v_peaks = (*it) -> dropletFinder<T>(seg_data, seg_mask, thr_low, thr_high, rad, dr);
	npeaks += v_peaks.size();
    }

    return _ndarrayOfPeakPars(npeaks);
  }

//--------------------
//--------------------

  template <typename T, unsigned NDim>
  ndarray<const float, 2>
  peakFinder( const ndarray<const T, NDim> data
            , const ndarray<const mask_t, NDim> mask
            , const T& thr
	    , const float& r0=5
            , const float& dr=0.05
            )
  {
    if(m_pbits & 256) MsgLog(_name(), info, "in peakFinder");

    if(! _initAlgImgProc<T,NDim>(data, mask)) { ndarray<const float, 2> empty; return empty; }

    unsigned npeaks = 0;

    for (std::vector<AlgImgProc*>::iterator it = v_algip.begin(); it != v_algip.end(); ++it) {

        size_t ind = (*it)->segind() * m_stride;              
	const ndarray<const T,2>      seg_data(&data.data()[ind], m_sshape);
	const ndarray<const mask_t,2> seg_mask(&m_mask[ind], m_sshape);

        std::vector<Peak>& v_peaks = (*it) -> peakFinder<T>(seg_data, seg_mask, thr, r0, dr);
	npeaks += v_peaks.size();
    }

    if(m_pbits & 256) MsgLog(_name(), info, "total number of peaks=" << npeaks);    

    return _ndarrayOfPeakPars(npeaks);
  }

//--------------------
//--------------------

};

} // namespace ImgAlgos

//#include "../src/AlgArrProc.cpp"

#endif // IMGALGOS_ALGARRPROC_H
