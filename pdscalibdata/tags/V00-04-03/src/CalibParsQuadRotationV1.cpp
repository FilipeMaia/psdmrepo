//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibParsQuadRotationV1...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/CalibParsQuadRotationV1.h"

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

CalibParsQuadRotationV1::CalibParsQuadRotationV1 ()
{
  m_quad_rotation[0] = 180;
  m_quad_rotation[1] =  90;
  m_quad_rotation[2] =   0;
  m_quad_rotation[3] = 270;
}

//----------------

CalibParsQuadRotationV1::CalibParsQuadRotationV1( const std::vector<double> v_parameters )
{
    if (v_parameters.size() != NUMBER_OF_PARAMETERS) {
        WithMsgLog("CalibParsQuadRotationV1", error, str) {
        str << "Expected number of parameters is " << NUMBER_OF_PARAMETERS ;
        str << ", read from file " << v_parameters.size() ;
        str << ": check the file.\n" ;
        }       
        abort();
    }
    size_t arr_size = sizeof( double ) * v_parameters.size();
    memcpy( &m_quad_rotation, &v_parameters[0], arr_size );
    //this->print();
}

void CalibParsQuadRotationV1::print()
{
  cout << endl << "QuadRotation:" << endl;  
  for( int quad=0; quad<NQuad; ++quad ) { cout << "  " << m_quad_rotation[quad]; }
  cout << endl;
}

//----------------
//--------------
// Destructor --
//--------------
CalibParsQuadRotationV1::~CalibParsQuadRotationV1 ()
{
}

} // namespace pdscalibdata
