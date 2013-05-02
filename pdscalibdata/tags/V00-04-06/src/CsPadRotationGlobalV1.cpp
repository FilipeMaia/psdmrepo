//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadRotationGlobalV1...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/CsPadRotationGlobalV1.h"

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
CsPadRotationGlobalV1::CsPadRotationGlobalV1 ()
{
  double arr[] = {  0,   0,  270,  270,  180,  180,  270,  270,
                   90,  90,    0,    0,  270,  270,    0,    0,
                  180, 180,   90,   90,    0,    0,   90,   90,
                  270, 270,  180,  180,   90,   90,  180,  180 };
  memcpy( &m_rotation, &arr[0], sizeof( double ) * NQuad * NSect );
}

//----------------

CsPadRotationGlobalV1::CsPadRotationGlobalV1( const std::vector<double> v_parameters )
{
    if (v_parameters.size() != NUMBER_OF_PARAMETERS) {
        WithMsgLog("CsPadRotationGlobalV1", error, str) {
        str << "Expected number of parameters is " << NUMBER_OF_PARAMETERS ;
        str << ", read from file " << v_parameters.size() ;
        str << ": check the file.\n" ;
        }       
        abort();
    }
    size_t arr_size = sizeof( double ) * v_parameters.size();
    memcpy( &m_rotation, &v_parameters[0], arr_size );
    //this->print();
}

void CsPadRotationGlobalV1::print()
{
  cout << endl << "Rotation global:" << endl;  
  for( int quad=0; quad<NQuad; ++quad ) {
    for( int sect=0; sect<NSect; ++sect ) { cout << "  " << m_rotation[quad][sect]; }
    cout << endl;
  }
}

//----------------
//--------------
// Destructor --
//--------------
CsPadRotationGlobalV1::~CsPadRotationGlobalV1 ()
{
}

} // namespace pdscalibdata
