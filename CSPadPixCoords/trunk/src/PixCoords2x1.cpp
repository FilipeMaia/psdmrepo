//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoords2x1...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadPixCoords/PixCoords2x1.h"

//-----------------
// C/C++ Headers --
//-----------------

#include <iostream> // for cout
//#include <fstream>

//#include <string>
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

namespace CSPadPixCoords {

//----------------
// Constructors --
//----------------

PixCoords2x1::PixCoords2x1 (float row_size_um, float col_size_um, float gap_row_size_um)
    : m_row_size_um(row_size_um)
    , m_col_size_um(col_size_um)
    , m_gap_row_size_um(gap_row_size_um)
{
  m_gap_size_um = 2 * m_gap_row_size_um - m_row_size_um; 

  fill_pix_coords_2x1();

  k_row_um_to_pix = 1./ m_row_size_um;
  k_col_um_to_pix = 1./ m_col_size_um;
  k_ort_um_to_pix = 1.;
}

//--------------
// Destructor --
//--------------

PixCoords2x1::~PixCoords2x1 ()
{
}

//--------------

void PixCoords2x1::fill_pix_coords_2x1()
{
  for (int col=0; col<NCols2x1; col++) m_coor_col[col] = col * m_col_size_um;
  for (int row=0; row<NRows2x1; row++) {
    m_coor_row[row] = (row<NRowsASIC) ? row * m_row_size_um : row * m_row_size_um + m_gap_size_um;
    for (int col=0; col<NCols2x1; col++) m_coor_ort[col][row] = 0;
  }

  m_coor_row_max = m_coor_row[NRows2x1-1];
  m_coor_col_max = m_coor_col[NCols2x1-1];
}

//--------------

void PixCoords2x1::print_member_data()
{
  cout << "PixCoords2x1::print_member_data():"      << endl
       << "NRows2x1          " << NRows2x1          << endl
       << "NCols2x1          " << NCols2x1          << endl
       << "NRowsASIC         " << NRowsASIC         << endl
       << "m_coor_row_max    " << m_coor_row_max    << endl
       << "m_coor_col_max    " << m_coor_col_max    << endl
       << "m_row_size_um     " << m_row_size_um     << endl
       << "m_col_size_um     " << m_col_size_um     << endl
       << "m_gap_row_size_um " << m_gap_row_size_um << endl
       << "m_gap_size_um     " << m_gap_size_um     << endl;
}

//--------------

void PixCoords2x1::print_selected_coords_2x1(ARRAXIS arraxis)
{
  string str_coord;
  switch (arraxis)
    {
    case ROW : str_coord = "ROW"; break;
    case COL : str_coord = "COL"; break;
    case ORT : str_coord = "ORT"; break;
    default  : return;
    }

  cout << "\nPixCoords2x1::print_selected_coords_2x1(" << str_coord << ")\n";
  for (int row=0; row<NRows2x1; row+=20) {
    for (int col=0; col<NCols2x1; col+=20) {

      switch (arraxis)
	{
        case ROW : cout << m_coor_row[row]      << "  "; break;
        case COL : cout << m_coor_col[col]      << "  "; break;
        case ORT : cout << m_coor_ort[col][row] << "  "; break;
        default  : continue;
	}
    }
    cout << endl;
  }
}

//--------------

float PixCoords2x1::getPixCoorRot000_um (COORDINATE icoor, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case X : return m_coor_row[row];
    case Y : return m_coor_col[col];
    case Z : return m_coor_ort[col][row];
    default: return 0;
    }
}

//--------------

float PixCoords2x1::getPixCoorRot090_um (COORDINATE icoor, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case X : return m_coor_col[col];                 
    case Y : return m_coor_row_max - m_coor_row[row];
    case Z : return m_coor_ort[col][row];	     
    default: return 0;
    }
}

//--------------

float PixCoords2x1::getPixCoorRot180_um (COORDINATE icoor, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case X : return m_coor_row_max - m_coor_row[row];
    case Y : return m_coor_col_max - m_coor_col[col];
    case Z : return m_coor_ort[col][row];
    default: return 0;
    }
}

//--------------

float PixCoords2x1::getPixCoorRot270_um (COORDINATE icoor, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case X : return m_coor_col_max - m_coor_col[col];
    case Y : return m_coor_row[row];
    case Z : return m_coor_ort[col][row];
    default: return 0;
    }
}

//--------------

float PixCoords2x1::getPixCoorRotN90_um (ORIENTATION n90, COORDINATE icoor, unsigned row, unsigned col)
{
  switch (n90)
    {
    case R000 : return getPixCoorRot000_um (icoor, row, col);
    case R090 : return getPixCoorRot090_um (icoor, row, col);
    case R180 : return getPixCoorRot180_um (icoor, row, col);
    case R270 : return getPixCoorRot270_um (icoor, row, col);
    default   : return 0;
    }
}

//--------------
//--------------

float PixCoords2x1::getPixCoorRot000_pix (COORDINATE icoor, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case X : return k_row_um_to_pix * m_coor_row[row];
    case Y : return k_col_um_to_pix * m_coor_col[col];
    case Z : return k_ort_um_to_pix * m_coor_ort[col][row];
    default: return 0;
    }
}

//--------------

float PixCoords2x1::getPixCoorRot090_pix (COORDINATE icoor, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case X : return k_col_um_to_pix * m_coor_col[col];                 
    case Y : return k_row_um_to_pix * (m_coor_row_max - m_coor_row[row]);
    case Z : return k_ort_um_to_pix * m_coor_ort[col][row];	     
    default: return 0;
    }
}

//--------------

float PixCoords2x1::getPixCoorRot180_pix (COORDINATE icoor, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case X : return k_row_um_to_pix * (m_coor_row_max - m_coor_row[row]);
    case Y : return k_col_um_to_pix * (m_coor_col_max - m_coor_col[col]);
    case Z : return k_ort_um_to_pix * m_coor_ort[col][row];
    default: return 0;
    }
}

//--------------

float PixCoords2x1::getPixCoorRot270_pix (COORDINATE icoor, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case X : return k_col_um_to_pix * (m_coor_col_max - m_coor_col[col]);
    case Y : return k_row_um_to_pix * m_coor_row[row];
    case Z : return k_ort_um_to_pix * m_coor_ort[col][row];
    default: return 0;
    }
}

//--------------

float PixCoords2x1::getPixCoorRotN90_pix (ORIENTATION n90, COORDINATE icoor, unsigned row, unsigned col)
{
  switch (n90)
    {
    case R000 : return getPixCoorRot000_pix (icoor, row, col);
    case R090 : return getPixCoorRot090_pix (icoor, row, col);
    case R180 : return getPixCoorRot180_pix (icoor, row, col);
    case R270 : return getPixCoorRot270_pix (icoor, row, col);
    default   : return 0;
    }
}

//--------------

float PixCoords2x1::getPixCoorRotN90 ( UNITS units, ORIENTATION n90, COORDINATE icoor, unsigned row, unsigned col)
{
  switch (units)
    {
    case  UM  : return getPixCoorRotN90_um  (n90, icoor, row, col);
    case  PIX : return getPixCoorRotN90_pix (n90, icoor, row, col);
    default   : return 0;
    }
}

//--------------

//--------------
//--------------

} // namespace CSPadPixCoords

//--------------
