//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibParsRotationV1...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/CalibParsRotationV1.h"

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
CalibParsRotationV1::CalibParsRotationV1 ()
{
}

//----------------

CalibParsRotationV1::CalibParsRotationV1( const std::vector<float> v_parameters )
{
    if (v_parameters.size() != NUMBER_OF_PARAMETERS) {
        WithMsgLog("CalibParsRotationV1", error, str) {
        str << "Expected number of parameters is " << NUMBER_OF_PARAMETERS ;
        str << ", read from file " << v_parameters.size() ;
        str << ": check the file.\n" ;
        }       
        abort();
    }
    size_t arr_size = sizeof( float ) * v_parameters.size();
    memcpy( &m_rotation, &v_parameters[0], arr_size );
    //this->print();
}

void CalibParsRotationV1::print()
{
  cout << endl << "Rotation:" << endl;  
  for( int quad=0; quad<NQuad; ++quad ) {
    for( int sect=0; sect<NSect; ++sect ) { cout << "  " << m_rotation[quad][sect]; }
    cout << endl;
  }
}

//----------------
//--------------
// Destructor --
//--------------
CalibParsRotationV1::~CalibParsRotationV1 ()
{
}

} // namespace pdscalibdata
