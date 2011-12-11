//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibParsOffsetV1...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/CalibParsOffsetV1.h"

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
CalibParsOffsetV1::CalibParsOffsetV1 ()
{
}

//----------------

CalibParsOffsetV1::CalibParsOffsetV1( const std::vector<double> v_parameters )
{
    if (v_parameters.size() != NUMBER_OF_PARAMETERS) {
        WithMsgLog("CalibParsOffsetV1", error, str) {
        str << "Expected number of parameters is " << NUMBER_OF_PARAMETERS ;
        str << ", read from file " << v_parameters.size() ;
        str << ": check the file.\n" ;
        }       
        abort();
    }
    size_t NPars    = NQuad;
    size_t arr_size = sizeof( double ) * v_parameters.size()/3;
    memcpy( &m_offset_x, &v_parameters[0],       arr_size );
    memcpy( &m_offset_y, &v_parameters[NPars],   arr_size );
    memcpy( &m_offset_z, &v_parameters[NPars*2], arr_size );
    //this->print();
}

void CalibParsOffsetV1::print()
{
  cout << endl;
  cout << "Quad offset X:"; for( int q=0; q<NQuad; ++q ) {cout << "  " << m_offset_x[q];} cout << endl;
  cout << "Quad offset Y:"; for( int q=0; q<NQuad; ++q ) {cout << "  " << m_offset_y[q];} cout << endl;
  cout << "Quad offset Z:"; for( int q=0; q<NQuad; ++q ) {cout << "  " << m_offset_z[q];} cout << endl;
}

//----------------

//--------------
// Destructor --
//--------------
CalibParsOffsetV1::~CalibParsOffsetV1 ()
{
}

} // namespace pdscalibdata
