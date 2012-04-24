#ifndef CSPADPIXCOORDS_IMAGE2D_H
#define CSPADPIXCOORDS_IMAGE2D_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Image2D.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace CSPadPixCoords {

/// @addtogroup CSPadPixCoords

/**
 *  @ingroup CSPadPixCoords
 *
 *  @brief Image2D class provides access to the 2D image data.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see CSPadImageProducer, PixCoordsTest
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

template <typename T>
class Image2D  {
public:

  // Default constructor
  Image2D () ;

  // Regular constructor
  /**
   *  @brief Stores the pointer to the 2D array containing image data and its sizes.
   *
   *  @param[in] data    Pointer to the 2D array of type T with image data.
   *  @param[in] nrows   Number of rows (1st index) in the 2D array.
   *  @param[in] ncols   Number of columns (2nd index) in the 2D array.   
   */
  Image2D (const T* data, size_t nrows, size_t ncols) ;

  // Destructor
  virtual ~Image2D () ;

  /**
   *  Methods of this class provide access to the 2D array and its 
   *  transformed versions after up-down, left-right flips, transpose, and
   *  rotations by n*90 degree.
   */
  T    getValue  (int row, int col);
  T    flipud    (int row, int col);
  T    fliplr    (int row, int col);
  T    transpose (int row, int col);
  T    rot000    (int row, int col);
  T    rot090    (int row, int col);
  T    rot180    (int row, int col);
  T    rot270    (int row, int col);
  T    rotN90    (int row, int col, int Nx90=0);

  /**
   *  @brief Returns the number of columns after rotations by n*90 degree.
   */
  size_t getNCols       (int Nx90=0);

  /**
   *  @brief Returns the number of rows after rotations by n*90 degree.
   */
  size_t getNRows       (int Nx90=0);

  /**
   *  @brief Returns pointer to data array
   */
  const T* data() { return m_data; }

  void printImage       (int Nx90=0);
  void printEntireImage (int Nx90=0);
  void saveImageInFile  (const std::string &fname, int Nx90=0);


private:

  // Copy constructor and assignment are disabled by default
  Image2D ( const Image2D& ) ;
  Image2D operator = ( const Image2D& ) ;

//------------------
// Static Members --
//------------------

  // Data members
  const T* m_data;
  size_t   m_nrows;
  size_t   m_ncols;
  size_t   m_nrows_transposed;
  size_t   m_ncols_transposed;
};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_IMAGE2D_H
