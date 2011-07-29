//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoordsCSPad...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadPixCoords/PixCoordsCSPad.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace CSPadPixCoords {

//----------------
// Constructors --
//----------------
PixCoordsCSPad::PixCoordsCSPad ( PixCoordsQuad *pix_coords_quad,  PSCalib::CSPadCalibPars *cspad_calibpar )
  : m_pix_coords_quad(pix_coords_quad)
  , m_cspad_calibpar (cspad_calibpar)
{
  cout << "PixCoordsQuad::PixCoordsCSPad:" << endl;
  //m_pix_coords_quad -> print_member_data(); 
  //m_cspad_calibpar  -> printCalibPars();
}

//--------------
// Destructor --
//--------------
PixCoordsCSPad::~PixCoordsCSPad ()
{
}

} // namespace CSPadPixCoords
