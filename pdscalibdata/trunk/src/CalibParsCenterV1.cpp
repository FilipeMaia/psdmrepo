//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibParsCenterV1...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/CalibParsCenterV1.h"

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
CalibParsCenterV1::CalibParsCenterV1 ()
{
}

//----------------

CalibParsCenterV1::CalibParsCenterV1( const std::vector<double> v_parameters )
{
    if (v_parameters.size() != NUMBER_OF_PARAMETERS) {
        WithMsgLog("CalibParsCenterV1", error, str) {
        str << "Expected number of parameters is " << NUMBER_OF_PARAMETERS ;
        str << ", read from file " << v_parameters.size() ;
        str << ": check the file.\n" ;
        }       
        abort();
    }
    size_t NPars = NQuad * NSect;
    size_t arr_size = sizeof( double ) * v_parameters.size()/3;
    memcpy( &m_center_x, &v_parameters[0],       arr_size );
    memcpy( &m_center_y, &v_parameters[NPars],   arr_size );
    memcpy( &m_center_z, &v_parameters[NPars*2], arr_size );
    //this->print();
}

void CalibParsCenterV1::print()
{
  cout << endl << "X:" << endl;  
  for( int quad=0; quad<NQuad; ++quad ) {
    for( int sect=0; sect<NSect; ++sect ) { cout << "  " << m_center_x[quad][sect]; }
    cout << endl;
  }
  cout << "Y:" << endl;  
  for( int quad=0; quad<NQuad; ++quad ) {
    for( int sect=0; sect<NSect; ++sect ) { cout << "  " << m_center_y[quad][sect]; }
    cout << endl;
  }
  cout << "Z:" << endl;  
  for( int quad=0; quad<NQuad; ++quad ) {
    for( int sect=0; sect<NSect; ++sect ) { cout << "  " << m_center_z[quad][sect]; }
    cout << endl;
  }
}

//----------------
//--------------
// Destructor --
//--------------
CalibParsCenterV1::~CalibParsCenterV1 ()
{
}

} // namespace pdscalibdata
