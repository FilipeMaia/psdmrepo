//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class QuadParameters...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadImage/QuadParameters.h"

//-----------------
// C/C++ Headers --
//-----------------

//#include <iostream> // for cout
//#include <fstream>

//#include <string>
using namespace std;

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

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

  QuadParameters::QuadParameters (uint32_t quadNumber, 
                                  std::vector<int> image_shape, 
                                  size_t nrows, 
                                  size_t ncols, 
                                  uint32_t numAsicsStored, 
                                  uint32_t roiMask) :

  m_quadNumber(quadNumber),
  v_image_shape(image_shape),
  m_nrows(nrows),
  m_ncols(ncols), 
  m_numAsicsStored(numAsicsStored),
  m_roiMask(roiMask)
{
  cout << "Here in QuadParameters::QuadParameters" << endl;
}

//----------------

void QuadParameters::print()
{
    WithMsgLog("QuadParameters::print()", info, str) {
      str << "\n m_nrows          = " << m_nrows;
      str << "\n m_ncols          = " << m_ncols;
      str << "\n m_quadNumber     = " << m_quadNumber;
      str << "\n m_numAsicsStored = " << m_numAsicsStored;
      str << "\n m_roiMask        = " << m_roiMask;
      str << "\n v_image_shape    = " << v_image_shape[0]  // Nquads
	                      << ", " << v_image_shape[1]  // Nrows
	                      << ", " << v_image_shape[2]; // Ncols
    }        
}

//--------------
// Destructor --
//--------------
QuadParameters::~QuadParameters ()
{
  //delete [] m_data; 
}
} // namespace CSPadImage
