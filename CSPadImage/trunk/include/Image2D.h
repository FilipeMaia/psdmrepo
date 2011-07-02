#ifndef CSPADIMAGE_IMAGE2D_H
#define CSPADIMAGE_IMAGE2D_H

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
class Image2D  {
public:

  // Default constructor
  Image2D () ;

  // Regular constructor
  Image2D (const T* data, size_t nrows, size_t ncols) ;

  // Regular constructor
  //Image2D (T* data, size_t nrows, size_t ncols) :
  //  m_data(data),
  //  m_nrows(nrows),
  //  m_ncols(ncols)
  //{}

  // Destructor
  virtual ~Image2D () ;

  void getValue (int row, int col, T &v);
  T    getValue (int row, int col);
  void printImage ();


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
};

} // namespace CSPadImage

#endif // CSPADIMAGE_IMAGE2D_H
