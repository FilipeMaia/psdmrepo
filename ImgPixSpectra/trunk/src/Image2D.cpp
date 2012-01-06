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
#include "ImgPixSpectra/Image2D.h"
//#include "ImgPixSpectra/ImageCSPad2x1.h"

//-----------------
// C/C++ Headers --
//-----------------

#include <iostream> // for cout
#include <fstream>

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

namespace ImgPixSpectra {

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
  //cout << "Here in Image2D<T>::Image2D" << endl;
    m_nrows_transposed = ncols;
    m_ncols_transposed = nrows;
}

//----------------
 
template <typename T>
inline
T Image2D<T>::getValue (int row, int col)
{
  return m_data[row*m_ncols + col];
  //return m_data->getValue (row, col);
}

//----------------

template <typename T>
inline
T Image2D<T>::flipud (int row, int col)
{
  return getValue(m_nrows-row-1, col);
}

//----------------

template <typename T>
inline
T Image2D<T>::fliplr (int row, int col)
{
  return getValue(row, m_ncols-col-1);
}

//----------------

template <typename T>
inline
T Image2D<T>::transpose (int row, int col)
{
  int col_transposed = row;
  int row_transposed = col;

  if (col < (int)m_ncols_transposed and row < (int)m_nrows_transposed ) 
    return getValue(row_transposed, col_transposed);
  else
    {
      cout << "Image2D<T>::transpose: "   
	   <<  "  row="   << row_transposed 
           <<  "  col="   << col_transposed
           <<  "  nrows=" << m_nrows_transposed
           <<  "  ncols=" << m_ncols_transposed
           << cout;
      return 0;
    }
}

//----------------

template <typename T>
inline
T Image2D<T>::rot000 (int row, int col) // fliplr (transpose)
{
  return getValue(row,col);
}

//----------------

template <typename T>
inline
T Image2D<T>::rot090 (int row, int col) // fliplr (transpose)
{
  int col_transposed = row;
  int row_transposed = col;
  return getValue(row_transposed, m_ncols-col_transposed-1);
}

//----------------

template <typename T>
inline
T Image2D<T>::rot270 (int row, int col) // flipud (transpose)
{
  int col_transposed = row;
  int row_transposed = col;
  return getValue(m_nrows-row_transposed-1, col_transposed);
}

//----------------

template <typename T>
inline
T Image2D<T>::rot180 (int row, int col) // flipud and fliplr
{
  return getValue(m_nrows-row-1, m_ncols-col-1);
}

//----------------

template <typename T>
inline
T Image2D<T>::rotN90 (int row, int col, int N) // Generalazed rotation by N*90 degree
{
       if (N==0) return rot000(row,col);
  else if (N==1) return rot090(row,col);
  else if (N==2) return rot180(row,col);
  else if (N==3) return rot270(row,col);
  else           return rot000(row,col);
}

//----------------

template <typename T>
inline
size_t Image2D<T>::getNCols(int Nx90)
{
  if ( (Nx90+2)%2 == 0 ) return m_ncols;
  else                   return m_nrows;
}

//----------------

template <typename T>
inline
size_t Image2D<T>::getNRows(int Nx90)
{
  if ( (Nx90+2)%2 == 0 ) return m_nrows;
  else                   return m_ncols;
}

//----------------
//----------------

template <typename T>
void Image2D<T>::printImage (int Nx90)
{
        cout << "Image2D<T>: Rotation by 90*" << Nx90 << "=" << Nx90*90 << " degree" << endl;

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
void Image2D<T>::printEntireImage (int Nx90)
{
        cout << "Image2D<T>: Rotation by 90*" << Nx90 << "=" << Nx90*90 << " degree" << endl;

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

//----------------

template <typename T>
void Image2D<T>::saveImageInFile (const std::string &fname, int Nx90)
{
    cout << "Image2D<T>::saveImageInFile: ";

    ofstream file; 
    file.open(fname.c_str(),ios_base::out);

        for (size_t row = 0; row < getNRows(Nx90); row++) {
          for (size_t col = 0; col < getNCols(Nx90); col++) {

            file << this->rotN90 (row,col,Nx90) << "  ";
          }
            file << endl;
        }

    file.close();
    cout << "The 2x1 image (ncols,nrows="
         << this->getNCols(Nx90) << ","
         << this->getNRows(Nx90)
         << " with rotation by 90*" << Nx90 << "=" << Nx90*90 << " degree)" 
         << " is saved in file " << fname << endl; 
}




//--------------
// Destructor --
//--------------
template <typename T>
Image2D<T>::~Image2D ()
{
  //  delete [] m_data; 
}


//-----------------------------------
// Instatiation of templated classes
//-----------------------------------

template class ImgPixSpectra::Image2D<uint16_t>;
template class ImgPixSpectra::Image2D<float>;
template class ImgPixSpectra::Image2D<int>;
  //template class ImgPixSpectra::Image2D<ImageCSPad2x1<uint16_t>>;

} // namespace ImgPixSpectra

//----------------
