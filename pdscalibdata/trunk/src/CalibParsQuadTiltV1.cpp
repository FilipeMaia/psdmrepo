//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibParsQuadTiltV1...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/CalibParsQuadTiltV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <string.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace pdscalibdata {

//----------------
// Constructors --
//----------------

CalibParsQuadTiltV1::CalibParsQuadTiltV1 ()
{
  std::fill_n(&m_quad_tilt[0], int(NQuad), double(0));
}

//----------------

CalibParsQuadTiltV1::CalibParsQuadTiltV1( const std::vector<double> v_parameters )
{
    if (v_parameters.size() != NUMBER_OF_PARAMETERS) {
        WithMsgLog("CalibParsQuadTiltV1", error, str) {
        str << "Expected number of parameters is " << NUMBER_OF_PARAMETERS ;
        str << ", read from file " << v_parameters.size() ;
        str << ": check the file.\n" ;
        }       
        abort();
    }
    size_t arr_size = sizeof( double ) * v_parameters.size();
    memcpy( &m_quad_tilt, &v_parameters[0], arr_size );
    //this->print();
}

void CalibParsQuadTiltV1::print()
{
  cout << endl << "QuadTilt:" << endl;  
  for( int quad=0; quad<NQuad; ++quad ) { cout << "  " << m_quad_tilt[quad]; }
  cout << endl;
}

//----------------
//--------------
// Destructor --
//--------------
CalibParsQuadTiltV1::~CalibParsQuadTiltV1 ()
{
}

} // namespace pdscalibdata
