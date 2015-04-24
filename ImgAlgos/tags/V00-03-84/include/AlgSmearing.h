#ifndef IMGALGOS_ALGSMEARING_H
#define IMGALGOS_ALGSMEARING_H

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
#include <iostream> // for cout
#include <cstddef>  // for size_t
#include <cstring>  // for memcpy

using namespace std;

//----------------------
// Base Class Headers --
//----------------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "MsgLogger/MsgLogger.h"
#include "ndarray/ndarray.h"

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
 *  @brief AlgSmearing - is a smearing algorithm for ndarray<T,2> using 2-d Gaussian weights.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 *
 *  @see AlgSmearing
 *
 *
 *  @anchor interface
 *  @par<interface> Interface Description
 *
 *
 * 
 *  @li  Include
 *  @code
 *  #include "ImgAlgos/AlgSmearing.h"
 *  #include "ndarray/ndarray.h"     // need it for I/O arrays
 *  @endcode
 *
 *
 *
 *  @li Initialization
 *  \n
 *  @code
 *  double      sigma    = 2.5;
 *  int         nsm      = 3;
 *  double      thr_low  = 10;
 *  unsigned    opt      = 1;
 *  unsigned    pbits    = 0177777;
 *  size_t      seg      = 2;
 *  size_t      rowmin   = 10;
 *  size_t      rowmax   = 170;
 *  size_t      colmin   = 100;
 *  size_t      colmax   = 200;
 *  double      value    = 0;
 * 
 *  AlgSmearing* p_sm = new AlgSmearing( sigma, nsm, thr_low, opt, pbits,
 *		      			 seg, rowmin, rowmax, colmin, colmax, value );
 *  @endcode
 *
 *
 *
 *  @li Do smearing
 *  @code
 *  ndarray<const T,2> nda_raw = ....;    // input raw ndarray
 *  ndarray<T,2> nda_sme(shape);          // output smeared ndarray
 *
 *  p_sm->smearing<T>(nda_raw, nda_sme);  // do smearing of input  ndarray<const T,2> nda_raw to output nda_sme
 *  @endcode
 *
 *
 *
 *  @li Access methods
 *  @code
 *  const size_t seg = p_sm->segind();
 *  @endcode
 *
 *
 *
 *  @li Print methods
 *  @code
 *  p_sm->printInputPars();
 *  p_sm->printWeights();
 *  @endcode
 */

//template <typename T>
class AlgSmearing  {
public:

  //static const size_t RSMMAX = 10;

  /**
   * @brief Class constructor is used for initialization of all paramaters. 
   * 
   * @param[in] sigma - smearing width parameter to generate the matrix of weights
   * @param[in] nsm   - (radial) number of neighbour rows and columns around pixel involved in smearing 
   * @param[in] thr_low - threshold on intensity; pixels with intensity above this threshold are accounted in smearing
   * @param[in] opt - pre-fill options for output array outside window region; 0-fill by zeros, 1-copy raw, 2-do nothing
   * @param[in] pbits  - print control bit-word; =0-print nothing, +1-input parameters, +2-matrix of weights, +128-detail at smearing.
   * @param[in] seg    - ROI segment index in the ndarray
   * @param[in] rowmin - ROI window limit
   * @param[in] rowmax - ROI window limit
   * @param[in] colmin - ROI window limit
   * @param[in] colmax - ROI window limit
   * @param[in] value  - pre-fill value 
   */

  AlgSmearing ( const double&   sigma   = 2
	      , const int&      nsm     = 5
              , const double&   thr_low = -1e10
              , const unsigned& opt     = 1
              , const unsigned& pbits   = 0
              , const size_t&   seg     = -1
              , const size_t&   rowmin  = 0
              , const size_t&   rowmax  = 1e6
              , const size_t&   colmin  = 0
              , const size_t&   colmax  = 1e6
              , const double&   value   = 0
              ) ;

  /// Destructor
  virtual ~AlgSmearing () {}

  /**
   * @brief Evaluates the matrix of weights distributed as 2-d Gaussian
   * 
   * If sigma=0 - matrix of weights is not used, but is filled out for print. 
   */

  void evaluateWeights();

  /// Prints the matrix of weights
  void printWeights();

  /// Prints memeber data
  void printInputPars();

  /**
   * @brief Get smearing weight
   * 
   * @param[in] dr - deviatin in rows from smeared pixel
   * @param[in] dc - deviatin in columns from smeared pixel
   */
  double weight(int dr, int dc) { return m_weights[abs(dr)][abs(dc)]; }

  /// Returns segment index in the ndarray
  const size_t& segind(){ return m_seg; }

  // Copy constructor and assignment are disabled by default
  AlgSmearing ( const AlgSmearing& ) ;
  AlgSmearing& operator = ( const AlgSmearing& ) ;


protected:

private:

  double   m_sigma;    // smearing sigma in pixel size
  int      m_nsm;      // number of pixels for smearing [i0-m_nsm, i0+m_nsm]
  int      m_nsm1;     // = m_nsm + 1
  double   m_thr_low;  // low threshold on pixel amplitude which will be involved in smearing
  unsigned m_opt;      // options for ndarray pre-fill
  unsigned m_pbits;    // pirnt control bit-word
  size_t   m_seg;      // segment index in the ndarray, <0 - for all segments
  size_t   m_rowmin;   // window for smearing
  size_t   m_rowmax;
  size_t   m_colmin;
  size_t   m_colmax;
  double   m_value;    // pre-fill and below threshold value

  ndarray<double,2> m_weights; //2-d array of weights

  /// Returns string name of the class for messanger
  std::string _name(){return std::string("ImgAlgos::AlgSmearing");}

//--------------------
  /**
   * @brief Returns smeared intensity of a single-pixel
   * 
   * @param[in] nda_raw - ndarray with raw intensities
   * @param[in] r0 - row of the smeared pixel
   * @param[in] c0 - column of the smeared pixel
   */

template <typename T>
double _smearPixAmp(const ndarray<const T,2>& nda_raw, const size_t& r0, const size_t& c0)
{
  const T *p_raw = nda_raw.data();
  size_t nrows = nda_raw.shape()[0];
  size_t ncols = nda_raw.shape()[1];

  double sum_aw = 0;
  double sum_w  = 0;
  double     w  = 0;
  unsigned ind  = 0;

  int rmin = std::max(0, int(r0-m_nsm));
  int rmax = std::min(nrows, r0+m_nsm+1);
  int cmin = std::max(0, int(c0-m_nsm));
  int cmax = std::min(ncols, c0+m_nsm+1);

  for (int r = rmin; r < rmax; r++) {
    for (int c = cmin; c < cmax; c++) {

      ind = r*ncols + c;
      w = weight(r-r0, c-c0);
      sum_w  += w;
      sum_aw += p_raw[ind] * w;

      //cout << "dr, dc, ind, w=" << r-r0 << " " << c-c0 << " " << ind << " " << w << endl;
    }
  }
  return (sum_w>0)? sum_aw / sum_w : m_value;
}




//--------------------

public:

//--------------------
  /**
   * @brief Smearing of the 2-d ndarray, one pass
   * 
   * @param[in]  nda_raw - ndarray with raw intensities
   * @param[out] nda_sme - ndarray with smeared intensities
   * 
   * If sigma=0 - smearing is turned off output array is a copy of input. 
   */

template <typename T>
void smearing(const ndarray<const T,2>& nda_raw, ndarray<T,2>& nda_sme)
{
  const T* p_raw = nda_raw.data();
  T*       p_sme = nda_sme.data();

  if(m_sigma == 0) { std::memcpy(p_sme, p_raw, nda_raw.size()*sizeof(T)); return; }

  T      thr   = (T) m_thr_low;
  T      val   = (T) m_value;
  size_t nrows = nda_raw.shape()[0];
  size_t ncols = nda_raw.shape()[1];

  size_t rmin = std::max(0, int(m_rowmin));
  size_t rmax = std::min(nrows, m_rowmax+1);
  size_t cmin = std::max(0, int(m_colmin));
  size_t cmax = std::min(ncols, m_colmax+1);

  if(m_pbits & 128) MsgLog(_name(), info, "  seg:"   << m_seg
            		               << "  nrows:" << nrows
            		               << "  ncols:" << ncols
            		               << "  rmin:"  << rmin
            		               << "  rmax:"  << rmax
            		               << "  cmin:"  << cmin
            		               << "  cmax:"  << cmax
            		               << "  nda_raw.size:"  << nda_raw.size()
           		               << "  nda_sme.size:"  << nda_sme.size()
			   );

  if     (m_opt==0) std::fill_n(p_sme, int(nda_raw.size()), T(m_value));
  else if(m_opt==1) std::memcpy(p_sme, p_raw, nda_raw.size()*sizeof(T));

  for (size_t r = rmin; r < rmax; r++) {
    for (size_t c = cmin; c < cmax; c++) {
      unsigned ind = r*ncols + c;
      p_sme[ind] = (p_raw[ind]>thr) ? (T)_smearPixAmp<T>(nda_raw, r, c) : val;
    }
  }
}

//--------------------

};

} // namespace ImgAlgos

#endif // IMGALGOS_ALGSMEARING_H
