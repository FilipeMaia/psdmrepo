//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Image2D...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadImage/Image2D.h"

//-----------------
// C/C++ Headers --
//-----------------

#include <iostream> // for cout

using namespace std;

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace CSPadImage {

//----------------
// Constructors --
//----------------

  // Default constructor
template <typename T>
Image2D<T>::Image2D (const T* data, size_t nrows, size_t ncols) :
    m_data(data),
    m_nrows(nrows),
    m_ncols(ncols)
{

  cout << "Here in Image2D<T>::Image2D" << endl;

}

 
template <typename T>
inline
void Image2D<T>::getValue (int row, int col, T &v)
{
  v = m_data[row*m_ncols + col];
}


template <typename T>
inline
T Image2D<T>::getValue (int row, int col)
{
  return m_data[row*m_ncols + col];
}


template <typename T>
void Image2D<T>::printImage ()
{
	for (size_t row = 0; row < m_nrows; row+=20) {
	  for (size_t col = 0; col < m_ncols; col+=20) {

	    //cout << m_data[row*m_ncols + col] << "  ";
	    cout << this->getValue(row,col) << "  ";

	  }
	    cout << endl;
	}
}




//--------------
// Destructor --
//--------------
template <typename T>
Image2D<T>::~Image2D ()
{
  //delete [] m_data; 
}


//-----------------------------------
// Instatiation of templated classes
//-----------------------------------

template class CSPadImage::Image2D<uint16_t>;

} // namespace CSPadImage
