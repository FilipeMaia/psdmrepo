#ifndef IMGALGOS_ALGIMGPROC_H
#define IMGALGOS_ALGIMGPROC_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description: see documentation below
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include <string>
#include <vector>
#include <iostream> // for cout, ostream
#include <cstddef>  // for size_t
#include <cstring>  // for memcpy
#include <cmath>    // for sqrt

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSCalib/CalibPars.h"  // for pixel_mask_t
#include "MsgLogger/MsgLogger.h"
#include "ndarray/ndarray.h"
#include "ImgAlgos/GlobalMethods.h"
#include "ImgAlgos/Window.h"


using namespace std;


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgAlgos {


/**
 * @brief Peak-work parameters
 */ 

struct PeakWork{
  unsigned  peak_npix;
  unsigned  peak_ireg;
  unsigned  peak_row;
  unsigned  peak_col;
  double    peak_amax;
  double    peak_atot;
  double    peak_ar1;
  double    peak_ar2;
  double    peak_ac1;
  double    peak_ac2;        
  unsigned  peak_rmin;
  unsigned  peak_rmax;
  unsigned  peak_cmin;
  unsigned  peak_cmax;   

  PeakWork( const unsigned& npix=0
	  , const unsigned& ireg=0
	  , const unsigned& row=0
	  , const unsigned& col=0
	  , const double&   amax=0
	  , const double&   atot=0
	  , const double&   ar1=0
	  , const double&   ar2=0
	  , const double&   ac1=0
	  , const double&   ac2=0        
	  , const unsigned& rmin=0
	  , const unsigned& rmax=0
	  , const unsigned& cmin=0
	  , const unsigned& cmax=0) :
      peak_npix(npix)
    , peak_ireg(ireg)
    , peak_row (row)
    , peak_col (col)
    , peak_amax(amax)
    , peak_atot(atot)
    , peak_ar1 (ar1)
    , peak_ar2 (ar2)
    , peak_ac1 (ac1)
    , peak_ac2 (ac2)        
    , peak_rmin(rmin)
    , peak_rmax(rmax)
    , peak_cmin(cmin)
    , peak_cmax(cmax)
    {}
};

/**
 * @brief Peak parameters
 */ 

struct Peak{
  float seg;
  float row;
  float col;
  float npix;
  float amp_max;
  float amp_tot;
  float row_cgrav; 
  float col_cgrav;
  float row_sigma;
  float col_sigma;
  float row_min;
  float row_max;
  float col_min;
  float col_max;
  float bkgd;
  float noise;
  float son;

  Peak& operator=(const Peak& rhs) {
    seg         = rhs.seg;
    row         = rhs.row;
    col         = rhs.col;
    npix        = rhs.npix;
    amp_max	= rhs.amp_max;
    amp_tot	= rhs.amp_tot;
    row_cgrav 	= rhs.row_cgrav;
    col_cgrav	= rhs.col_cgrav;
    row_sigma	= rhs.row_sigma;
    col_sigma	= rhs.col_sigma;
    row_min	= rhs.row_min;
    row_max	= rhs.row_max;
    col_min	= rhs.col_min;
    col_max	= rhs.col_max;
    bkgd	= rhs.bkgd;
    noise	= rhs.noise;
    son         = rhs.son;
    return *this;
  }
};

/// Stream insertion operator,
std::ostream& 
operator<<( std::ostream& os, const Peak& p);

/*
struct TwoIndexes {
  int i;
  int j;
};
*/

/**
 * @brief Structure to hold SoN (S/N) algorithm results
 */ 

struct SoNResult {
  double avg; // average intensity in the ring
  double rms; // rms in the ring
  double sig; // raw-avg
  double son; // sig/rms

  SoNResult(const double& av=0, const double& rm=0, const double& sg=0, const double& sn=0) :
    avg(av), rms(rm), sig(sg), son(sn) {}

  SoNResult& operator=(const SoNResult& rhs) {
    avg = rhs.avg;
    rms = rhs.rms;
    sig = rhs.sig;
    son = rhs.son;
    return *this;
  }
};

/// @addtogroup ImgAlgos

/**
 *  @ingroup ImgAlgos
 *
 *  @brief AlgImgProc - class for 2-d image processing algorithms.
 *
 *  This class is not suppose to be used separately. 
 *  Class AlgImgProc is a part of the Python-C++ algorithm inteface project.
 *
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 *
 *  @see AlgArrProc, pyImgAlgos.cpp, PyAlgos.py
 *
 *
 *  @anchor interface
 *  @par<interface> Interface Description
 *
 * 
 *  @li  Includes and typedefs
 *  @code
 *  #include "ImgAlgos/AlgImgProc.h"
 *  #include "ndarray/ndarray.h"     // need it for I/O arrays
 *  
 *  typedef ImgAlgos::AlgImgProc::conmap_t conmap_t;
 *  @endcode
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
 *  unsigned    pbits  = 0;  // 0-print nothing, 2-input parameters and S/N matrix of indexes, 512-tracking.
 * 
 *  ImgAlgos::AlgImgProc* aip = new ImgAlgos::AlgImgProc(seg, rowmin, rowmax, colmin, colmax, pbits);
 *  @endcode
 *
 *
 *  @li Define input parameters
 *  @code
 *  ndarray<const T,2> data = ....;    // calibrated data ndarray
 *  ndarray<mask_t,2>  mask = ....;    // mask ndarray, may be omitted
 *  ndarray<mask_t,2>  son;            // output S/N ndarray
 *  ndarray<const wind_t,2> winds = ((0,  0, 185,   0, 388), \
 *                                   (1, 10, 103,  10, 204), \
 *                                   (1, 10, 103, 250, 380));
 *  float       r0 = 5;
 *  float       dr = 0.05;
 *  ...
 *  @endcode
 *
 *
 *  @li Set methods
 *  @code
 *  aip->setSoNPars(r0,dr);
 *  aip->setWindows(winds);
 *  aip->setPeakSelectionPars(npix_min, npix_max, amax_thr, atot_thr, son_min);
 *  @endcode
 *
 *
 *  @li Get methods
 *  @code
 *   size_t ind = aip->segind()
 *   size_t counter = aip -> numberOfPixAboveThr<T>(seg_data, seg_mask, thr);
 *   double intensity = aip -> intensityOfPixAboveThr<T>(seg_data, seg_mask, thr);
 *   std::vector<Peak>& peaks = aip -> dropletFinder<T>(seg_data, seg_mask, thr_low, thr_high, rad, dr);
 *   std::vector<Peak>& peaks = aip -> peakFinder<T>(seg_data, seg_mask, thr, r0, dr);
 *   std::vector<Peak>& peaks = aip -> getVectorOfSelectedPeaks();
 *   std::vector<Peak>& peaks = aip -> getVectorOfPeaks();
 *
 *   // call after peakFinder ONLY!
 *   ndarray<conmap_t, 2>& conmap = aip -> mapOfConnectedPixels();
 *  @endcode
 *
 *
 *  @li Print methods
 *  @code
 *  aip->printInputPars();
 *  aip->printMatrixOfRingIndexes();
 *  aip->printVectorOfRingIndexes();
 *
 *  Peak& peak = ...
 *  cout << peak ...
 *  @endcode
 */

class AlgImgProc {
public:

  typedef unsigned shape_t;
  typedef PSCalib::CalibPars::pixel_mask_t mask_t;
  typedef uint32_t conmap_t;
  typedef PSCalib::CalibPars::pixel_status_t  pixel_status_t;   // uint16_t pixel_status_t;
  typedef float son_t;

  /**
   * @brief Class constructor is used for initialization of all paramaters. 
   * 
   * @param[in] seg    - ROI segment index in the ndarray
   * @param[in] rowmin - ROI window limit
   * @param[in] rowmax - ROI window limit
   * @param[in] colmin - ROI window limit
   * @param[in] colmax - ROI window limit
   * @param[in] pbits  - print control bit-word; =0-print nothing, +1-input parameters, +2-algorithm area, +128-all details.
   */

  AlgImgProc( const size_t&   seg     =-1
	    , const size_t&   rowmin  = 0
	    , const size_t&   rowmax  = 1e6
	    , const size_t&   colmin  = 0
	    , const size_t&   colmax  = 1e6
	    , const unsigned& pbits   = 0
	    ) ;

  /// Destructor
  virtual ~AlgImgProc () {}

  /// Prints memeber data
  void printInputPars();

  /// Set segment index in the >2-d ndarray
  void setSegment(const size_t& seg=-1){ m_seg = seg; }

  /**
   * @brief Set median (S/N) algorithm parameters
   * @param[in] r0 - radial parameter of the ring for S/N evaluation algorithm
   * @param[in] dr - ring width for S/N evaluation algorithm 
   */
  void setSoNPars(const float& r0=5, const float& dr=0.05);

  /// Set peak selection parameters
  void setPeakSelectionPars(const float& npix_min=2, 
                            const float& npix_max=200, 
                            const float& amax_thr=0, 
                            const float& atot_thr=0,
                            const float& son_min=3);

  /// Returns reference to Window object
  const Window& window(){ return m_win; }

  /// Returns segment index in the ndarray
  const size_t& segind(){ return m_seg; }

  /// Returns vector of all found peaks for this segment/window
  std::vector<Peak>& getVectorOfPeaks(){ return v_peaks; }

  /// Returns vector of selected peaks for this segment/window
  std::vector<Peak>& getVectorOfSelectedPeaks(){ return v_peaks_sel; }

  /// Returns map of connected pixels after peakFinder(.)
  ndarray<conmap_t, 2>& mapOfConnectedPixels() { return m_conmap; }

  /// Prints indexes
  void printMatrixOfRingIndexes();
  void printVectorOfRingIndexes();

  // Copy constructor and assignment are disabled by default
  AlgImgProc ( const AlgImgProc& ) ;
  AlgImgProc& operator = ( const AlgImgProc& ) ;

private:

  unsigned m_pbits;    // pirnt control bit-word
  size_t   m_seg;      // segment index (for ndarray with ndim>2)

  Window   m_win;      // work area window

  bool     m_init_son_is_done; // for S/N algorithm
  bool     m_use_mask;

  unsigned m_numreg;

  float    m_r0;       // radial parameter of the ring for S/N evaluation algorithm
  float    m_dr;       // ring width for S/N evaluation algorithm 

  SoNResult m_sonres_def;

  float    m_peak_npix_min; // peak selection parameter
  float    m_peak_npix_max; // peak selection parameter
  float    m_peak_amax_thr; // peak selection parameter
  float    m_peak_atot_thr; // peak selection parameter
  float    m_peak_son_min;  // peak selection parameter

  //Peak     m_peak;

  ndarray<pixel_status_t, 2> m_pixel_status;
  ndarray<conmap_t, 2>       m_conmap;
  std::vector<PeakWork>      v_peaks_work;
  std::vector<Peak>          v_peaks;
  std::vector<Peak>          v_peaks_sel;
  std::vector<TwoIndexes>    v_indexes;

  //ndarray<Peak, 1>           m_peaks;
  // mask_t*                    m_seg_mask_def;


  /// Returns string name of the class for messanger
  inline const char* _name() {return "ImgAlgos::AlgImgProc";}

  //std::vector<TwoIndexes> v_indexes;

  /// Recursive method which checks m_pixel_status[r][c] and numerates connected regions in m_conmap[r][c].
  void _findConnectedPixels(const unsigned& r, const unsigned& c);

  /// Makes m_conmap - map of connected pixels with enumerated regions from m_pixel_status and counts m_numreg
  void _makeMapOfConnectedPixels();

  /// Decide whether PeakWork should be processed and included in the v_peaks
  bool _peakWorkIsPreSelected(const PeakWork& pw);

  /// Decide if peak should be processed or not and included in the v_peaks 
  bool _peakIsPreSelected(const Peak& peak);

  /// Decide if peak should be included or not in the output v_peaks
  bool _peakIsSelected(const Peak& peak);

  /// Makes vector of peaks v_peaks from v_peaks_work
  void _makeVectorOfPeaks();

  /// Makes vector of selected peaks v_peaks_sel from v_peaks
  void _makeVectorOfSelectedPeaks();

  /// Evaluate ring indexes for median algorithm
  void _evaluateRingIndexes(const float& r0, const float& dr);


//--------------------
  /**
   * @brief Makes m_pixel_status array by setting to 1/0 all good-above-threshold/bad pixels
   * 
   * @param[in]  data - ndarray with calibrated intensities
   * @param[in]  mask - ndarray with mask of bad/good (0/1) pixels
   * @param[in]  thr  - threshold on data values
   */

template <typename T>
void
_makeMapOfPixelStatus( const ndarray<const T,2>&      data
                     , const ndarray<const mask_t,2>& mask
                     , const T& thr
                     )
{
  if(m_pbits & 512) MsgLog(_name(), info, "in _makeMapOfPixelStatus, seg=" << m_seg << " thr=" << thr);
  if(m_pbits & 512) m_win.print();

  m_pixel_status = make_ndarray<pixel_status_t>(data.shape()[0], data.shape()[1]);
  for(unsigned r = m_win.rowmin; r<m_win.rowmax; r++) {
    for(unsigned c = m_win.colmin; c<m_win.colmax; c++) {
      m_pixel_status[r][c] = (mask[r][c] && (data[r][c]>thr)) ? 1 : 0;
    }
  }
}

//--------------------
  /**
   * @brief Process data ndarray using map of connected pixels m_conmap 
   * and collect peak information in std::vector<PeakWork> v_peaks_work 
   * 
   * @param[in]  data - ndarray with calibrated intensities
   */

template <typename T>
void
_procConnectedPixels(const ndarray<const T,2>& data)
{
  if(m_pbits & 512) MsgLog(_name(), info, "in _procConnectedPixels, seg=" << m_seg);
  //if(m_pbits & 512) m_win.print();

  v_peaks_work.clear();

  if(m_numreg==0) return;

  v_peaks_work.reserve(m_numreg+1); // use numeration from 1
  PeakWork pw0; // def init with 0
  pw0.peak_cmin = m_win.colmax;
  pw0.peak_rmin = m_win.rowmax;
  std::fill_n(&v_peaks_work[0], int(m_numreg+1), pw0);

  for(unsigned r = m_win.rowmin; r<m_win.rowmax; r++) {
    for(unsigned c = m_win.colmin; c<m_win.colmax; c++) {
      unsigned ireg = m_conmap[r][c];
      if(! ireg) continue;

      //std::cout << " reg=" << ireg;

      double amp = (double)data[r][c];

      PeakWork& pw = v_peaks_work[ireg];
      pw.peak_npix ++;   
      pw.peak_ireg = ireg;
      pw.peak_atot += amp;   
      pw.peak_ar1  += amp*r;    
      pw.peak_ar2  += amp*r*r;    
      pw.peak_ac1  += amp*c;    
      pw.peak_ac2  += amp*c*c;    

      if(amp > pw.peak_amax) {
        pw.peak_row  = r;
	pw.peak_col  = c;
        pw.peak_amax = amp;   
      }

      if(c < pw.peak_cmin) pw.peak_cmin = c;
      if(c > pw.peak_cmax) pw.peak_cmax = c;
      if(r < pw.peak_rmin) pw.peak_rmin = r;
      if(r > pw.peak_rmax) pw.peak_rmax = r;
    }
  }
}

//--------------------
  /**
   * @brief Loops over list of peaks m_peaks, evaluates SoN info and adds it to each peak.
   * 
   * @param[in]  data - ndarray with calibrated intensities
   * @param[in]  mask - ndarray with mask of bad/good (0/1) pixels
   * @param[in]  r0   - radial parameter of the ring for S/N evaluation algorithm
   * @param[in]  dr   - ring width for S/N evaluation algorithm
   */

template <typename T>
void _addSoNToPeaks( const ndarray<const T,2>& data
                   , const ndarray<const mask_t,2>& mask
	           , const float r0 = 5
	           , const float dr = 0.05
                   )
{
  if(m_pbits & 512) MsgLog(_name(), info, "in _addSoNToPeaks, seg=" << m_seg);

  setSoNPars(r0, dr);

  //ndarray<Peak, 1>m_peaks;
  //ndarray<Peak, 1>::iterator it;
  std::vector<Peak>::iterator it;

  for(it=v_peaks.begin(); it!=v_peaks.end(); ++it) { 
    Peak& peak = (*it);
    
    SoNResult sonres = evaluateSoNForPixel<T>((unsigned) peak.row, (unsigned) peak.col, data, mask);

    peak.bkgd  = sonres.avg;
    peak.noise = sonres.rms;
    peak.son   = sonres.son;
  }
}

//--------------------
  /**
   * @brief _procDroplet - process a single droplet candidate
   * 
   * @param[in]  data - ndarray with calibrated intensities
   * @param[in]  mask - ndarray with mask of bad/good (0/1) pixels
   * @param[in]  thr_high  - threshold on pixel intensity to be a candidate to "droplet"
   * @param[in]  rad -  
   * @param[in]  r0 - droplet central pixel row-coordinate 
   * @param[in]  c0 - droplet central pixel column-coordinate   
   */

template <typename T>
void
_procDroplet( const ndarray<const T,2>&      data
            , const ndarray<const mask_t,2>& mask
            , const T& thr_low
            , const unsigned& rad
            , const unsigned& r0
            , const unsigned& c0
            )
{
  if(m_pbits & 512) MsgLog(_name(), info, "in _procDroplet, seg=" << m_seg << " r0=" << r0 << " c0=" << c0);

  double a0 = data[r0][c0];
  unsigned npix = 0;
  double   samp = 0;
  double   sac1 = 0;
  double   sac2 = 0;
  double   sar1 = 0;
  double   sar2 = 0;

  unsigned rmin = std::max((int)m_win.rowmin, int(r0-rad));
  unsigned rmax = std::min((int)m_win.rowmax, int(r0+rad+1));
  unsigned cmin = std::max((int)m_win.colmin, int(c0-rad));
  unsigned cmax = std::min((int)m_win.colmax, int(c0+rad+1));

  for(unsigned r = rmin; r<rmax; r++) {
    for(unsigned c = cmin; c<cmax; c++) {
      double a = data[r][c];

      if(a>a0) return;  // This is not a local maximum inside rad...

      if(mask[r][c] && a>thr_low) {
	npix += 1;
	samp += a;
	sar1 += a*r;
	sac1 += a*c;
	sar2 += a*r*r;
	sac2 += a*c*c;
      }
    }
  }

  Peak peak;

  peak.seg       = m_seg;
  peak.row       = r0;
  peak.col       = c0;
  peak.npix      = npix;
  peak.amp_max   = a0;
  peak.amp_tot   = samp;
  peak.row_cgrav = sar1/samp;
  peak.col_cgrav = sac1/samp;
  peak.row_sigma = (npix>1) ? std::sqrt( sar2/samp - peak.row_cgrav * peak.row_cgrav ) : 0;
  peak.col_sigma = (npix>1) ? std::sqrt( sac2/samp - peak.col_cgrav * peak.col_cgrav ) : 0;
  peak.row_min   = rmin;
  peak.row_max   = rmax;
  peak.col_min   = cmin;
  peak.col_max   = cmax;  
  peak.bkgd      = 0; //sonres.avg;
  peak.noise     = 0; //sonres.rms;
  peak.son       = 0; //sonres.son;

  if(_peakIsPreSelected(peak)) v_peaks.push_back(peak);
}
//--------------------
  /**
   * @brief dropletFinder - two-threshold peak finding algorithm in the region defined by the radial parameter
   * 
   * @param[in]  data - ndarray with calibrated intensities
   * @param[in]  mask - ndarray with mask of bad/good (0/1) pixels
   * @param[in]  thr_low   - threshold on pixel intensity to be considered in this algorithm 
   * @param[in]  thr_high  - threshold on pixel intensity to be a candidate to "droplet"
   * @param[in]  rad       - radius in pixels of squared region to find droplet relative to central pixel
   */
template <typename T>
void 
_makeVectorOfDroplets( const ndarray<const T,2>&      data
                     , const ndarray<const mask_t,2>& mask
                     , const T& thr_low
                     , const T& thr_high
                     , const unsigned& rad=5
                     )
{
  if(m_pbits & 512) MsgLog(_name(), info, "in dropletFinder, seg=" << m_seg);

  v_peaks.clear(); // this vector will be filled out for each window

  for(unsigned r = m_win.rowmin; r<m_win.rowmax; r++)
    for(unsigned c = m_win.colmin; c<m_win.colmax; c++)
      if(mask[r][c] && (data[r][c]>thr_high)) 
        _procDroplet<T>(data,mask,thr_low,rad,r,c);	
}

//--------------------
//--------------------
//--------------------
//--------------------

public:

//--------------------
  /**
   * @brief numberOfPixAboveThr - counts a number of pixels above threshold
   * 
   * @param[in]  data - ndarray with calibrated intensities
   * @param[in]  mask - ndarray with mask of bad/good (0/1) pixels
   * @param[in]  thr  - threshold on data values
   */

template <typename T>
unsigned
numberOfPixAboveThr( const ndarray<const T,2>&      data
                   , const ndarray<const mask_t,2>& mask
                   , const T& thr
                   )
{
  if(m_pbits & 512) MsgLog(_name(), info, "in numberOfPixAboveThr, seg=" << m_seg);

  m_win.validate(data.shape());

  unsigned npix = 0;
  for(unsigned r = m_win.rowmin; r<m_win.rowmax; r++) {
    for(unsigned c = m_win.colmin; c<m_win.colmax; c++) {
      if(mask[r][c] && (data[r][c]>thr)) npix++;
    }
  }
  return npix;
}

//--------------------
  /**
   * @brief intensityOfPixAboveThr - evaluates total intensity of pixels above threshold
   * 
   * @param[in]  data - ndarray with calibrated intensities
   * @param[in]  mask - ndarray with mask of bad/good (0/1) pixels
   * @param[in]  thr  - threshold on data values
   */

template <typename T>
double
intensityOfPixAboveThr( const ndarray<const T,2>&      data
                      , const ndarray<const mask_t,2>& mask
                      , const T& thr
                      )
{
  if(m_pbits & 512) MsgLog(_name(), info, "in intensityOfPixAboveThr, seg=" << m_seg);

  m_win.validate(data.shape());

  double amptot = 0;
  for(unsigned r = m_win.rowmin; r<m_win.rowmax; r++) {
    for(unsigned c = m_win.colmin; c<m_win.colmax; c++) {
      if(mask[r][c] && (data[r][c]>thr)) amptot += (double)data[r][c];
    }
  }
  return amptot;
}

//--------------------
  /**
   * @brief dropletFinder - two-threshold peak finding algorithm in the region defined by the radial parameter
   * 
   * @param[in]  data - ndarray with calibrated intensities
   * @param[in]  mask - ndarray with mask of bad/good (0/1) pixels
   * @param[in]  thr_low   - threshold on pixel intensity to be considered in this algorithm 
   * @param[in]  thr_high  - threshold on pixel intensity to be a candidate to "droplet"
   * @param[in]  rad       - radius in pixels of squared region to find droplet relative to central pixel
   * @param[in]  dr        - width of the ring of radius rad for SoN algorithm
   */

template <typename T>
std::vector<Peak>&
dropletFinder( const ndarray<const T,2>&      data
             , const ndarray<const mask_t,2>& mask
             , const T& thr_low
             , const T& thr_high
             , const unsigned& rad=5
             , const float&    dr=0.05
             )
{
  if(m_pbits & 512) MsgLog(_name(), info, "in dropletFinder, seg=" << m_seg);

  m_win.validate(data.shape());

  _makeVectorOfDroplets(data, mask, thr_low, thr_high, rad);
  _addSoNToPeaks<T>(data, mask, float(rad), dr);
  _makeVectorOfSelectedPeaks();
  return v_peaks_sel; 
}

//--------------------

//--------------------
  /**
   * @brief peakFinder - makes a list of peaks for groups of connected pixels above threshold.
   * 1) uses data and mask and finds groups of connected pixels with innensity above threshold;
   * 2) each group of connected pixels is processed as a single peak,
   *    its parameters and correlators are collected in struct PeakWork,
   *    all of them are collected in the std::vector<PeakWork> v_peaks_work;
   * 3) v_peaks_work is processed and the list of peak parameters is saved in std::vector<PeakWork> v_peaks_work.
   * 
   * @param[in]  data - ndarray with calibrated intensities
   * @param[in]  mask - ndarray with mask of bad/good (0/1) pixels
   * @param[in]  thr  - threshold on data values
   * @param[in]  r0   - radius for SoN algorithm
   * @param[in]  dr   - width of the ring of radius rad for SoN algorithm
   */

template <typename T>
std::vector<Peak>&
peakFinder( const ndarray<const T,2>&      data
          , const ndarray<const mask_t,2>& mask
          , const T& thr
	  , const float r0 = 5
	  , const float dr = 0.05
          )
{
  if(m_pbits & 512) MsgLog(_name(), info, "in peakFinder, seg=" << m_seg);

  m_win.validate(data.shape());

  _makeMapOfPixelStatus<T>(data, mask, thr);
  _makeMapOfConnectedPixels();
  _procConnectedPixels<T>(data);
  _makeVectorOfPeaks();
  _addSoNToPeaks<T>(data, mask, r0, dr);
  _makeVectorOfSelectedPeaks();

  return v_peaks_sel; 
}

//--------------------
  /**
   * @brief Evaluate per-pixel result of the S/N (median) algorithm to data using mask
   * 
   * S/N is evaluated for any pixel specified by the (row,col).  
   * If mask is provided, and pixel is masked (0) then default result is returned.
   * S/N algorithm uses non-masked surrounding pixels in the ring m_r0, m_dr.
   * Thresholds are not applied in order to prevent offset of the average value of the background level.
   * 
   * @param[in]  row  - pixel row
   * @param[in]  col  - pixel column
   * @param[in]  data - ndarray with calibrated intensities
   * @param[in]  mask - ndarray with mask of bad/good (0/1) pixels
   */

template <typename T>
SoNResult
evaluateSoNForPixel( const unsigned& row
                   , const unsigned& col
                   , const ndarray<const T,2>& data
                   , const ndarray<const mask_t,2>& mask
                   )
{
  //if(m_pbits & 512) MsgLog(_name(), info, "in evaluateSoNForPixel, seg=" << m_seg << " row=" << row << ", col=" << col);

  // S/N algorithm initialization
  if(! m_init_son_is_done) {
    _evaluateRingIndexes(m_r0, m_dr);
    m_win.validate(data.shape());
    m_use_mask = (mask.empty()) ? false : true;
    m_init_son_is_done = true;
  }

  if(m_use_mask && (!mask[row][col])) return m_sonres_def;
  //if(m_use_mask && (!mask[row][col])) return SoNResult({});

  double   amp  = 0;
  unsigned sum0 = 0;
  double   sum1 = 0;
  double   sum2 = 0;

  for(vector<TwoIndexes>::const_iterator ij  = v_indexes.begin();
                                         ij != v_indexes.end(); ij++) {
    int ir = row + (ij->i);
    int ic = col + (ij->j);

    if(ic < (int)m_win.colmin || ic > (int)m_win.colmax) continue;
    if(ir < (int)m_win.rowmin || ir > (int)m_win.rowmax) continue;
    if(m_use_mask && (! mask[ir][ic])) continue;

    amp = (double)data[ir][ic];
    sum0 ++;
    sum1 += amp;
    sum2 += amp*amp;
  }
  //SoNResult res = m_sonres_def;
  SoNResult res;

  if(sum0) {
    res.avg = sum1/sum0;                              // Averaged background level
    res.rms = std::sqrt(sum2/sum0 - res.avg*res.avg); // RMS os the background around peak
    res.sig = data[row][col]      - res.avg;          // Signal above the background
    if (res.rms>0) res.son = res.sig/res.rms;         // S/N ratio
  }

  return res;
}

//--------------------
  /**
   * @brief Get ALL results of the S/N algorithm applied to data ndarray using mask
   * 
   * @param[in]  data   - ndarray with calibrated intensities
   * @param[in]  mask   - ndarray with mask of bad/good (0/1) pixels
   * @param[out] result - ndarray with results of median algorithm (average bkgd, rms, signal, S/N)
   * @param[in]  do_fill_def - pre-fill ndarray with default values
   */

template <typename T>
void getSoNResult( const ndarray<const T,2>& data
                 , const ndarray<const mask_t,2>& mask
                 , ndarray<SoNResult,2>& result
                 , const bool& do_fill_def = false
                 )
{
  if(m_pbits & 512) MsgLog(_name(), info, "in getSoNResult, seg=" << m_seg);

  if(do_fill_def) std::fill_n(&result, int(data.size()), m_sonres_def);

  for(unsigned r = m_win.rowmin; r<m_win.rowmax; r++) {
    for(unsigned c = m_win.colmin; c<m_win.colmax; c++) {
      result[r][c] = evaluateSoNForPixel<T>(r, c, data, mask);
    }
  }
}

//--------------------
  /**
   * @brief Get ONLY S/N for data ndarray using mask
   * 
   * @param[in]  data   - ndarray with calibrated intensities
   * @param[in]  mask   - ndarray with mask of bad/good (0/1) pixels
   * @param[out] son - ndarray with results of median algorithm (average bkgd, rms, signal, S/N)
   * @param[in]  do_fill_def - pre-fill ndarray with default values
   */

template <typename T>
void getSoN( const ndarray<const T,2>& data
           , const ndarray<const mask_t,2>& mask
           , ndarray<son_t,2>& son
           , const bool& do_fill_def = false
           )
{
  if(m_pbits & 512) MsgLog(_name(), info, "in getSoN, seg=" << m_seg);

  if(do_fill_def) std::fill_n(&son, int(data.size()), son_t(0));

  for(unsigned r = m_win.rowmin; r<m_win.rowmax; r++) {
    for(unsigned c = m_win.colmin; c<m_win.colmax; c++) {
      son[r][c] = evaluateSoNForPixel<T>(r, c, data, mask).son;
    }
  }
}

//--------------------
//--------------------
//--------------------

};

} // namespace ImgAlgos

#endif // IMGALGOS_ALGIMGPROC_H
