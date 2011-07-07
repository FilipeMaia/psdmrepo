//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImageCSPad2x1...
//
// Author List:
//      Mikhail S. Dubrovin
//
// Image of the 2x1 have a gap comparing to the original array.
// This gap is accounted in getValue.
// This is the only principal difference with class Image2D.
// Member data are also extended to account for this difference.
// In this class we assume that the image has fixed size 388x185 with gap=3
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadImage/ImageCSPad2x1.h"

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

template <typename T>
ImageCSPad2x1<T>::ImageCSPad2x1 (const T* data, size_t gap_ncols, size_t nrows, size_t ncols) :
  m_data(data)
{
  m_gap_ncols  = gap_ncols;
  m_nrows_arr  = nrows;
  m_ncols_arr  = ncols; 
  m_nrows      = m_nrows_arr;
  m_ncols      = m_ncols_arr+gap_ncols;
  m_ncols_arr_half          = m_ncols_arr/2;
  m_ncols_arr_half_plus_gap = m_ncols_arr_half + gap_ncols;

  cout << "Here in ImageCSPad2x1<T>::ImageCSPad2x1" << endl;
}

//----------------

template <typename T>
inline
T ImageCSPad2x1<T>::getValue (size_t row, size_t col) 
{
  if (row<m_nrows) {

    if       (col<m_ncols_arr_half)          { return m_data[row*m_ncols_arr + col]; }
    else if  (col<m_ncols_arr_half_plus_gap) { return 0; }
    else if  (col<m_ncols)                   { return m_data[row*m_ncols_arr + col - m_gap_ncols]; }
    else { cout << "ImageCSPad2x1<T>: COLUMN INDEX " << col 
                << " EXCEEDS MAXIMAL EXPECTED NUMBER " << m_ncols_arr 
                << "+" << m_gap_ncols << endl; return 0; }
  }
  else { cout   << "ImageCSPad2x1<T>: ROW INDEX " << row 
                << " EXCEEDS MAXIMAL EXPECTED NUMBER "<< m_nrows << endl; return 0; }
}

//----------------

template <typename T>
inline
T ImageCSPad2x1<T>::rot000 (int row, int col) // fliplr (transpose)
{
  return getValue(row,col);
}

//----------------

template <typename T>
inline
T ImageCSPad2x1<T>::rot090 (int row, int col) // fliplr (transpose)
{
  int col_transposed = row;
  int row_transposed = col;
  return getValue(row_transposed, m_ncols-col_transposed-1);
}

//----------------

template <typename T>
inline
T ImageCSPad2x1<T>::rot270 (int row, int col) // flipud (transpose)
{
  int col_transposed = row;
  int row_transposed = col;
  return getValue(m_nrows-row_transposed-1, col_transposed);
}

//----------------

template <typename T>
inline
T ImageCSPad2x1<T>::rot180 (int row, int col) // flipud and fliplr
{
  return getValue(m_nrows-row-1, m_ncols-col-1);
}

//----------------

template <typename T>
inline
T ImageCSPad2x1<T>::rotN90 (int row, int col, int Nx90) // Generalazed rotation by N*90 degree
{
       if (Nx90==0) return rot000(row,col);
  else if (Nx90==1) return rot090(row,col);
  else if (Nx90==2) return rot180(row,col);
  else if (Nx90==3) return rot270(row,col);
  else              return rot000(row,col);
}

//----------------

template <typename T>
inline
size_t ImageCSPad2x1<T>::getNCols(int Nx90)
{
  if ( (Nx90+2)%2 == 0 ) return m_ncols;
  else                   return m_nrows;
}

//----------------

template <typename T>
inline
size_t ImageCSPad2x1<T>::getNRows(int Nx90)
{
  if ( (Nx90+2)%2 == 0 ) return m_nrows;
  else                   return m_ncols;
}

//----------------

template <typename T>
void ImageCSPad2x1<T>::printImage (int Nx90)
{
        cout << "ImageCSPad2x1<T>: Rotation by 90*" << Nx90 << "=" << Nx90*90 << " degree" << endl;

	cout << "   ncols=" << this->getNCols(Nx90)
	     << "   nrows=" << this->getNRows(Nx90)
             << endl;

	for (size_t row = 0; row < getNRows(Nx90); row+=20) {
	  for (size_t col = 0; col < getNCols(Nx90); col+=20) {

	    cout << this->rotN90 (row,col,Nx90) << "  ";
	  }
	    cout << endl;
	}
}

//----------------

template <typename T>
void ImageCSPad2x1<T>::printEntireImage (int Nx90)
{
        cout << "ImageCSPad2x1<T>: Rotation by 90*" << Nx90 << "=" << Nx90*90 << " degree" << endl;

	cout << "   ncols=" << this->getNCols(Nx90)
	     << "   nrows=" << this->getNRows(Nx90)
             << endl;

        for (size_t row = 0; row < getNRows(Nx90); row++) {
	  for (size_t col = 0; col < getNCols(Nx90); col++) {

	    cout << this->rotN90 (row,col,Nx90) << "  ";
	  }
	    cout << endl;
	}
}

//--------------
// Destructor --
//--------------
template <typename T>
ImageCSPad2x1<T>::~ImageCSPad2x1 ()
{
  delete [] m_data; 
}

//-----------------------------------
// Instatiation of templated classes
//-----------------------------------

template class CSPadImage::ImageCSPad2x1<uint16_t>;

} // namespace CSPadImage
