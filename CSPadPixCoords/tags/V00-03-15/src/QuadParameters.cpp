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
#include "CSPadPixCoords/QuadParameters.h"

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

namespace CSPadPixCoords {

//----------------
// Constructors --
//----------------

  QuadParameters::QuadParameters (uint32_t quadNumber, 
                                  size_t nrows, 
                                  size_t ncols, 
                                  uint32_t numAsicsStored, 
                                  uint32_t roiMask) :

  m_quadNumber(quadNumber),
  m_nrows(nrows),
  m_ncols(ncols), 
  m_numAsicsStored(numAsicsStored),
  m_roiMask(roiMask)
{
  //cout << "Here in QuadParameters::QuadParameters" << endl;
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
    }        
}

//--------------
// Destructor --
//--------------
QuadParameters::~QuadParameters ()
{
  //delete [] m_data; 
}
} // namespace CSPadPixCoords
