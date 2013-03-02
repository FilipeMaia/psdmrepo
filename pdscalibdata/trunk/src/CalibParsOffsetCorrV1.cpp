//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibParsOffsetCorrV1...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/CalibParsOffsetCorrV1.h"

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
CalibParsOffsetCorrV1::CalibParsOffsetCorrV1 ()
{
  std::fill_n(&m_offset_corr_x[0], int(NQuad), double(0));
  std::fill_n(&m_offset_corr_y[0], int(NQuad), double(0));
  std::fill_n(&m_offset_corr_z[0], int(NQuad), double(0));
}
//----------------

CalibParsOffsetCorrV1::CalibParsOffsetCorrV1( const std::vector<double> v_parameters )
{
    if (v_parameters.size() != NUMBER_OF_PARAMETERS) {
        WithMsgLog("CalibParsOffsetCorrV1", error, str) {
        str << "Expected number of parameters is " << NUMBER_OF_PARAMETERS ;
        str << ", read from file " << v_parameters.size() ;
        str << ": check the file.\n" ;
        }       
        abort();
    }
    size_t NPars    = NQuad;
    size_t arr_size = sizeof( double ) * v_parameters.size()/3;
    memcpy( &m_offset_corr_x, &v_parameters[0],       arr_size );
    memcpy( &m_offset_corr_y, &v_parameters[NPars],   arr_size );
    memcpy( &m_offset_corr_z, &v_parameters[NPars*2], arr_size );
    //this->print();
}

void CalibParsOffsetCorrV1::print()
{
  cout << endl;
  cout << "Quad offset correction X:"; for( int q=0; q<NQuad; ++q ) {cout << "  " << m_offset_corr_x[q];} cout << endl;
  cout << "Quad offset correction Y:"; for( int q=0; q<NQuad; ++q ) {cout << "  " << m_offset_corr_y[q];} cout << endl;
  cout << "Quad offset correction Z:"; for( int q=0; q<NQuad; ++q ) {cout << "  " << m_offset_corr_z[q];} cout << endl;
}

//----------------

//--------------
// Destructor --
//--------------
CalibParsOffsetCorrV1::~CalibParsOffsetCorrV1 ()
{
}

} // namespace pdscalibdata
