#ifndef PSCALIB_GLOBALMETHODS_H
#define PSCALIB_GLOBALMETHODS_H

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// $Revision$
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <string>
//#include <vector>
//#include <map>
//#include <fstream>  // open, close etc.

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"
#include <cstddef>  // for size_t

//-----------------------------

namespace PSCalib {

/// @addtogroup PSCalib PSCalib
/**
 *  @ingroup PSCalib
 *
 *  @brief module GlobalMethods.h has Global Methods
 *
 */

//-----------------------------

static const size_t N2X1    = 2;
static const size_t ROWS2X1 = 185;
static const size_t COLS2X1 = 388;
static const size_t SIZE2X1 = COLS2X1*ROWS2X1; 
static const size_t SIZE2X2 = N2X1*SIZE2X1; 

//-----------------------------
/**
 * @brief Converts cspad2x2 ndarray data2x2[185,388,2] to two2x1[2,185,388] 
 * 
 * @param[in]  data2x2 - input ndarray shaped as [185,388,2]
 */
  template <typename T>
  ndarray<const T, 3> 
  data2x2ToTwo2x1(const ndarray<const T, 3>& data2x2)
  {
    ndarray<T, 3> two2x1 = make_ndarray<T>(N2X1, ROWS2X1, COLS2X1);
    
    for(size_t n=0; n<N2X1;    ++n) {
    for(size_t c=0; c<COLS2X1; ++c) {
    for(size_t r=0; r<ROWS2X1; ++r) {

      two2x1[n,r,c] = data2x2[r,c,n];  

    }
    }
    }
    return two2x1;
  }


//-----------------------------
/**
 * @brief Converts cspad2x2 ndarray two2x1[2,185,388] to data2x2[185,388,2]
 * 
 * @param[in]  two2x1 - input ndarray shaped as [2,185,388]
 */
  template <typename T>
  ndarray<const T, 3> 
  two2x1ToData2x2(const ndarray<const T, 3>& two2x1)
  {
    ndarray<T, 3> data2x2 = make_ndarray<T>(N2X1, ROWS2X1, COLS2X1);
    
    for(size_t n=0; n<N2X1;    ++n) {
    for(size_t c=0; c<COLS2X1; ++c) {
    for(size_t r=0; r<ROWS2X1; ++r) {

      data2x2[r,c,n] = two2x1[n,r,c];  

    }
    }
    }
    return data2x2;
  }

//-----------------------------
/**
 * @brief Converts cspad2x2 ndarray two2x1[2,185,388] to data2x2[185,388,2]
 * 
 * @param[in]  two2x1 - input ndarray shaped as [2,185,388]
 */
  template <typename T>
  void two2x1ToData2x2(T* A)
  {
    unsigned int shape_in [3] = {N2X1, ROWS2X1, COLS2X1};
    unsigned int shape_out[3] =       {ROWS2X1, COLS2X1, N2X1};

    ndarray<T, 3> two2x1(A, shape_in);
    ndarray<T, 3> data2x2(shape_out);

    for(size_t n=0; n<N2X1;    ++n) {
    for(size_t c=0; c<COLS2X1; ++c) {
    for(size_t r=0; r<ROWS2X1; ++r) {

      data2x2[r][c][n] = two2x1[n][r][c];

    }
    }
    }
    std::memcpy(A, data2x2.data(), data2x2.size()*sizeof(T));
  }

//-----------------------------

} // namespace PSCalib
//-----------------------------

#endif // PSCALIB_GLOBALMETHODS_H
