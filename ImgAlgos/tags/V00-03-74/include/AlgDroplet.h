#ifndef IMGALGOS_ALGDROPLET_H
#define IMGALGOS_ALGDROPLET_H

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
#include <cstddef>  // for size_t
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
 *  @brief Droplet(peak) finding algorithm which works in ROI window on const ndarray<const T,2> 
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

//template <typename T>
class AlgDroplet  {
public:

  static const size_t NDROPLETSBLOCK = 100;

  struct Droplet{
    double x;
    double y; 
    double ampmax;
    double amptot;
    unsigned npix;
  };
   // ? double s1;
   // ? double s2; 
   // ? double tilt_angle;  


  /**
   * @brief Constructor
   * 
   * @param[in] radius - (radial) number of neighbour rows and columns around pixel involved in droplet finding
   * @param[in] thr_low - threshold on intensity; pixels with intensity above this threshold are accounted in droplet formation
   * @param[in] thr_high - threshold on intensity; pixels with intensity above this threshold are considered as a droplet
   * @param[in] pbits  - print control bit-word
   * @param[in] seg    - ROI segment index in the ndarray
   * @param[in] rowmin - ROI window limit
   * @param[in] rowmax - ROI window limit
   * @param[in] colmin - ROI window limit
   * @param[in] colmax - ROI window limit
   */

  AlgDroplet (  const int&      radius   = 5
              , const double&   thr_low  = -1e10
              , const double&   thr_high = -1e10
              , const unsigned& pbits    = 0
              , const size_t&   seg      = -1
              , const size_t&   rowmin   = 0
              , const size_t&   rowmax   = 1e6
              , const size_t&   colmin   = 0
              , const size_t&   colmax   = 1e6
              ) ;

  /// Destructor
  virtual ~AlgDroplet () {}

  /**
   * @brief Evaluates the matrix of weights distributed as 2-d Gaussian
   * 
   * If sigma=0 - matrix of weights is not used, but is filled out for print. 
   */

  /// Prints memeber data
  void printInputPars();

  /// Prints vector of droplets
  void printDroplets();

  /// Returns segment index in the ndarray
  const size_t& segind(){ return m_seg; }

  // Copy constructor and assignment are disabled by default
  AlgDroplet ( const AlgDroplet& ) ;
  AlgDroplet& operator = ( const AlgDroplet& ) ;


protected:

private:

  int      m_radius;   // (radial) number of pixels for smearing [i0-m_radius, i0+m_radius]
  double   m_thr_low;  // low threshold on pixel amplitude which are accounted in droplet formation
  double   m_thr_high; // high threshold on pixel amplitude which are considered as a droplet
  unsigned m_pbits;    // pirnt control bit-word
  size_t   m_seg;      // segment index in the ndarray, <0 - for all segments
  size_t   m_rowmin;   // limits on the ROI window where algorithm is applied
  size_t   m_rowmax;
  size_t   m_colmin;
  size_t   m_colmax;

  std::vector<Droplet> v_droplets;

  /// Saves droplet information in the vector
  void saveDropletInfo(size_t& row, size_t& col, double& amp, double& amp_tot, unsigned& npix );

  /// Prints droplet information
  void printDropletInfo(const Droplet& d);

  /// Returns string with droplet parameters
  std::string strDropletPars(const Droplet& d);

  /// Returns reference to the accumulated vector of droplets
  const std::vector<Droplet>& getDroplets() { return v_droplets; }

  /// Returns string name of the class for messanger
  std::string name(){return std::string("ImgAlgos::AlgDroplet");}

//--------------------
  /**
   * @brief Checks if the current pixel has absolutly maximal intensity in the region defined by radius 
   * 
   * @param[in] nda - ndarray which is used to search for "droplets"
   * @param[in] r0 - row of the smeared pixel
   * @param[in] c0 - column of the smeared pixel
   */

template <typename T>
void _checkIfPixIsDroplet( const ndarray<const T,2>& nda, size_t r0, size_t c0 )
{
  size_t nrows = nda.shape()[0];
  size_t ncols = nda.shape()[1];

  //double   a0    = nda[r0*ncols + c0];
  double   a0    = nda[r0][c0];
  double   a     = 0;
  double   sum_a = 0;
  unsigned n_pix = 0;

  size_t rmin = std::max(0, int(r0-m_radius));
  size_t rmax = std::min(nrows, r0+m_radius+1);
  size_t cmin = std::max(0, int(c0-m_radius));
  size_t cmax = std::min(ncols, c0+m_radius+1);

  for (size_t r = rmin; r < rmax; r++) {
    for (size_t c = cmin; c < cmax; c++) {

      //a = nda[r*ncols + c];
      a = nda[r][c];

      if( a > a0 ) return; // This is not a local droplet...

      if( a > m_thr_low ) {
        sum_a += a;
        n_pix += 1;
      }
    }
  }

  // Save the droplet info here ----------
  saveDropletInfo(r0, c0, a0, sum_a, n_pix);

}

//--------------------
  /**
   * @brief Finds and accumulates droplets
   * @param[in] nda - ndarray to search for "droplets"
   */

template <typename T>
bool findDroplets( const ndarray<const T,2>& nda )
{
  v_droplets.clear();
  v_droplets.reserve(NDROPLETSBLOCK);

  const T* p_nda = nda.data();
  size_t   nrows = nda.shape()[0];
  size_t   ncols = nda.shape()[1];

  // Check thet ROI limits are consistent with image size
  size_t rmin = std::max(0, int(m_rowmin));
  size_t rmax = std::min(nrows, m_rowmax+1);
  size_t cmin = std::max(0, int(m_colmin));
  size_t cmax = std::min(ncols, m_colmax+1);

  T thr_high = (T) m_thr_high;

  // Find droplets in the ROI window
  for (size_t r = rmin; r < rmax; r++) {
    size_t ind = r*ncols + cmin;
    for (size_t c = cmin; c < cmax; c++, ind++) {
      if (p_nda[ind] > thr_high) _checkIfPixIsDroplet(nda,r,c);
    }
  }

  if(m_pbits & 2) MsgLog(name(), info, "Found number of droplets: " << (int)v_droplets.size() ) ;

  return (v_droplets.size()) ? true : false;
}

//--------------------
//--------------------

};

} // namespace ImgAlgos

#endif // IMGALGOS_ALGDROPLET_H
