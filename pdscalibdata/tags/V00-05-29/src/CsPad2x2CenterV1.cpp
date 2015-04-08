//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2CenterV1...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/CsPad2x2CenterV1.h"

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
CsPad2x2CenterV1::CsPad2x2CenterV1 ()
{
  double arr_x[] = {198., 198.};
  double arr_y[] = { 95., 308.};
 
  int NPars = NSect;
  size_t  arr_size = sizeof( double ) * NPars;
  memcpy     ( &m_center_x[0], arr_x, arr_size);
  memcpy     ( &m_center_y[0], arr_y, arr_size);
  std::fill_n( &m_center_z[0], NPars, double(0));
}

//----------------

CsPad2x2CenterV1::CsPad2x2CenterV1( const std::vector<double> v_parameters )
{
    if (v_parameters.size() != NUMBER_OF_PARAMETERS) {
        WithMsgLog("CsPad2x2CenterV1", error, str) {
        str << "Expected number of parameters is " << NUMBER_OF_PARAMETERS ;
        str << ", read from file " << v_parameters.size() ;
        str << ": check the file.\n" ;
        }       
        abort();
    }
    size_t NPars = NSect;
    size_t arr_size = sizeof( double ) * v_parameters.size()/3;
    memcpy( &m_center_x, &v_parameters[0],       arr_size );
    memcpy( &m_center_y, &v_parameters[NPars],   arr_size );
    memcpy( &m_center_z, &v_parameters[NPars*2], arr_size );
    //this->print();
}

void CsPad2x2CenterV1::print()
{
    cout << endl << "X:";  
    for( int sect=0; sect<NSect; ++sect ) { cout << "  " << m_center_x[sect]; }
    cout << endl;

    cout << "Y:";  
    for( int sect=0; sect<NSect; ++sect ) { cout << "  " << m_center_y[sect]; }
    cout << endl;

    cout << "Z:";  
    for( int sect=0; sect<NSect; ++sect ) { cout << "  " << m_center_z[sect]; }
    cout << endl;
}

//----------------
//--------------
// Destructor --
//--------------
CsPad2x2CenterV1::~CsPad2x2CenterV1 ()
{
}

} // namespace pdscalibdata
