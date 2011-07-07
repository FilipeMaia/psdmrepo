#ifndef CSPADIMAGE_IMAGECSPAD2X1_H
#define CSPADIMAGE_IMAGECSPAD2X1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImageCSPad2x1.
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

namespace CSPadImage {

/**
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

template <typename T>
class ImageCSPad2x1  {
public:

  // Default constructor
  ImageCSPad2x1 () ;

  // Regular constructor
  ImageCSPad2x1 (const T* data, size_t gap_ncols=3, size_t nrows=185, size_t ncols=388) ;

  // Destructor
  virtual ~ImageCSPad2x1 () ;

  T    getValue  (size_t  row, size_t  col);
  T    rot000    (int row, int col);
  T    rot090    (int row, int col);
  T    rot180    (int row, int col);
  T    rot270    (int row, int col);
  T    rotN90    (int row, int col, int Nx90);

  size_t getNCols(int Nx90);
  size_t getNRows(int Nx90);
  //T& operator[](size_t &index);
  void printImage       (int Nx90=0);
  void printEntireImage (int Nx90=0);


private:

  // Copy constructor and assignment are disabled by default
  ImageCSPad2x1 ( const ImageCSPad2x1& ) ;
  ImageCSPad2x1 operator = ( const ImageCSPad2x1& ) ;

//------------------
// Static Members --
//------------------

  // Data members
  const T* m_data;
  size_t   m_nrows;
  size_t   m_ncols;
  size_t   m_nrows_arr;
  size_t   m_ncols_arr;
  size_t   m_gap_ncols;
  size_t   m_ncols_arr_half;
  size_t   m_ncols_arr_half_plus_gap;
};

} // namespace CSPadImage

#endif // CSPADIMAGE_IMAGECSPAD2X1_H
