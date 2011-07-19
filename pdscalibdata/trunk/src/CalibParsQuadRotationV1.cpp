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

namespace pdscalibdata {

//----------------
// Constructors --
//----------------

CalibParsQuadRotationV1::CalibParsQuadRotationV1 ()
{
}

//----------------

CalibParsQuadRotationV1::CalibParsQuadRotationV1( const std::vector<float> v_parameters )
{
    if (v_parameters.size() != NUMBER_OF_PARAMETERS) {
        WithMsgLog("CalibParsQuadRotationV1", error, str) {
        str << "Expected number of parameters is " << NUMBER_OF_PARAMETERS ;
        str << ", read from file " << v_parameters.size() ;
        str << ": check the file.\n" ;
        }       
        abort();
    }
    size_t arr_size = sizeof( float ) * v_parameters.size();
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
