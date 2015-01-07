//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadCenterGlobalV1...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/CsPadCenterGlobalV1.h"

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
CsPadCenterGlobalV1::CsPadCenterGlobalV1 ()
{
  double arr_x[] = { 473.38,  685.26,  155.01,  154.08,  266.81,   53.95,  583.04,  582.15,
                     989.30,  987.12, 1096.93,  884.11, 1413.16, 1414.94, 1500.83, 1288.02,
                    1142.59,  930.23, 1459.44, 1460.67, 1347.57, 1559.93, 1032.27, 1033.44,
		     626.78,  627.42,  516.03,  729.15,  198.28,  198.01,  115.31,  327.66};
		                                                                          
  double arr_y[] = {1028.07, 1026.28, 1139.46,  926.91, 1456.78, 1457.35, 1539.71, 1327.89,
                    1180.51,  967.36, 1497.74, 1498.54, 1385.08, 1598.19, 1069.65, 1069.93,
                     664.89,  666.83,  553.60,  765.91,  237.53,  236.06,  152.17,  365.47,
                     510.38,  722.95,  193.33,  193.41,  308.04,   95.25,  625.28,  624.14};

  int NPars = NQuad * NSect;
  size_t  arr_size = sizeof( double ) * NPars;
  memcpy     ( &m_center_x[0][0], arr_x, arr_size);
  memcpy     ( &m_center_y[0][0], arr_y, arr_size);
  std::fill_n( &m_center_z[0][0], NPars, double(0));
}

//----------------

CsPadCenterGlobalV1::CsPadCenterGlobalV1( const std::vector<double> v_parameters )
{
    if (v_parameters.size() != NUMBER_OF_PARAMETERS) {
        WithMsgLog("CsPadCenterGlobalV1", error, str) {
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

void CsPadCenterGlobalV1::print()
{
  cout << endl << "Center global:" << endl;  
  cout << "X:" << endl;  
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
CsPadCenterGlobalV1::~CsPadCenterGlobalV1 ()
{
}

} // namespace pdscalibdata
