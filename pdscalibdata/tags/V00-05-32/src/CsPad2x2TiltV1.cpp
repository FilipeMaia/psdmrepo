//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2TiltV1...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/CsPad2x2TiltV1.h"

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

CsPad2x2TiltV1::CsPad2x2TiltV1 ()
{
  // Default tilts are 0:
  std::fill_n(&m_tilt[0], int(NSect), double(0));
}

//----------------

CsPad2x2TiltV1::CsPad2x2TiltV1( const std::vector<double> v_parameters )
{
    if (v_parameters.size() != NUMBER_OF_PARAMETERS) {
        WithMsgLog("CsPad2x2TiltV1", error, str) {
        str << "Expected number of parameters is " << NUMBER_OF_PARAMETERS ;
        str << ", read from file " << v_parameters.size() ;
        str << ": check the file.\n" ;
        }       
        abort();
    }
    size_t arr_size = sizeof( double ) * v_parameters.size();
    memcpy( &m_tilt, &v_parameters[0], arr_size );
    //this->print();
}

void CsPad2x2TiltV1::print()
{
    cout << endl << "Tilt:";  
    for( int sect=0; sect<NSect; ++sect ) { cout << "  " << m_tilt[sect]; }
    cout << endl;
}

//----------------

CsPad2x2TiltV1::~CsPad2x2TiltV1 ()
{
}

} // namespace pdscalibdata
