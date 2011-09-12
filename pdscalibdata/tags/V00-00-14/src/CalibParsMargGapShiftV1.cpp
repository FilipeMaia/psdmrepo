//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibParsMargGapShiftV1...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/CalibParsMargGapShiftV1.h"

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
CalibParsMargGapShiftV1::CalibParsMargGapShiftV1 ()
{
}

//----------------

CalibParsMargGapShiftV1::CalibParsMargGapShiftV1( const std::vector<double> v_parameters )
{
    if (v_parameters.size() != NUMBER_OF_PARAMETERS) {
        WithMsgLog("CalibParsMargGapShiftV1", error, str) {
        str << "Expected number of parameters is " << NUMBER_OF_PARAMETERS ;
        str << ", read from file " << v_parameters.size() ;
        str << ": check the file.\n" ;
        }       
        abort();
    }
    m_quad_marg_x  = v_parameters[0];
    m_quad_marg_y  = v_parameters[4];
    m_quad_marg_z  = v_parameters[8];

    m_marg_x       = v_parameters[1];
    m_marg_y       = v_parameters[5];
    m_marg_z       = v_parameters[9];

    m_gap_x        = v_parameters[2];
    m_gap_y        = v_parameters[6];
    m_gap_z        = v_parameters[10];

    m_shift_x      = v_parameters[3];
    m_shift_y      = v_parameters[7];
    m_shift_z      = v_parameters[11];

    //this->print();
}

void CalibParsMargGapShiftV1::print()
{
  cout << endl 
       << "Quad margine X,Y,Z:"
       << "  " << m_quad_marg_x
       << "  " << m_quad_marg_y
       << "  " << m_quad_marg_z
       << endl;

  cout << "Margine      X,Y,Z:"   
       << "  " << m_marg_x
       << "  " << m_marg_y
       << "  " << m_marg_z
       << endl;

  cout << "Gap          X,Y,Z:"   
       << "  " << m_gap_x
       << "  " << m_gap_y
       << "  " << m_gap_z
       << endl;

  cout << "Shift        X,Y,Z:"   
       << "  " << m_shift_x
       << "  " << m_shift_y
       << "  " << m_shift_z
       << endl;
}

//----------------


//--------------
// Destructor --
//--------------
CalibParsMargGapShiftV1::~CalibParsMargGapShiftV1 ()
{
}

} // namespace pdscalibdata
