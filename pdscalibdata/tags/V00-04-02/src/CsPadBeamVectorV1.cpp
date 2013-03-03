//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadBeamVectorV1...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/CsPadBeamVectorV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <string.h>
#include <stdlib.h>

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

CsPadBeamVectorV1::CsPadBeamVectorV1 ()
{
  //WithMsgLog("CsPadBeamVectorV1", warning, str) {
  //str << "Use defauld initialization to (0,0,0). CSPAD geometry IS NOT CORRECT!!!\n" 
  //    << "Provide calibration file <run-range>.data in expected place under the calib directoy" ;
  //}       
  std::fill_n(&m_beam_vector[0], int(NUMBER_OF_PARAMETERS), double(0));
}

//----------------

CsPadBeamVectorV1::CsPadBeamVectorV1( const std::vector<double> v_parameters )
{
    if (v_parameters.size() != NUMBER_OF_PARAMETERS) {
        WithMsgLog("CsPadBeamVectorV1", error, str) {
        str << "Expected number of parameters is " << NUMBER_OF_PARAMETERS ;
        str << ", read from file " << v_parameters.size() ;
        str << ": check the file.\n" ;
        }       
        abort();
    }
    size_t arr_size = sizeof( double ) * v_parameters.size();
    memcpy( &m_beam_vector, &v_parameters[0], arr_size );
    //this->print();
}

void CsPadBeamVectorV1::print()
{
  cout << endl << "Beam Vector:" << endl;  
  for( int i=0; i<NUMBER_OF_PARAMETERS; ++i ) { cout << "  " << m_beam_vector[i]; }
  cout << endl;
}

//----------------
//--------------
// Destructor --
//--------------
CsPadBeamVectorV1::~CsPadBeamVectorV1 ()
{
}

} // namespace pdscalibdata
