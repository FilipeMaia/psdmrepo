//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadCalibPars...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadImage/CSPadCalibPars.h"

//-----------------
// C/C++ Headers --
//-----------------

#include <iostream> // for cout
//#include <fstream>

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

namespace CSPadImage {

//----------------
// Constructors --
//----------------

CSPadCalibPars::CSPadCalibPars ( const std::string &xtc_file_name )
{
  cout << "CSPadCalibPars::CSPadCalibPars" 
       << "\nHere we have to find from the xtc_file_name the run number and find the calib directory..." << endl;

    m_expdir    = "/reg/d/psdm/CXI/cxi35711"; 
    m_calibdir  = "calib";
    m_calibtype = "CsPad::CalibV1"; 
    m_calibsrc  = "CxiDs1.0:Cspad.0"; 
    m_filename  = "1-end.data";

    // Temporary staff in:
    m_calibdir  = "/reg/neh/home/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi35711-r0009-det";

    v_calibname.push_back("center");
    v_calibname.push_back("center_corr");
    v_calibname.push_back("marg_gap_shift");
    v_calibname.push_back("offset");
    v_calibname.push_back("offset_corr");
    v_calibname.push_back("rotation");
    v_calibname.push_back("tilt");
    v_calibname.push_back("quad_rotation");
    v_calibname.push_back("quad_tilt");


    loadCalibPars ();
}

//----------------

void CSPadCalibPars::loadCalibPars ()
{
    for( vector<std::string>::const_iterator iterCalibName  = v_calibname.begin();
                                             iterCalibName != v_calibname.end(); iterCalibName++ )
      {
        m_cur_calibname = *iterCalibName;

	openCalibFile();
	readCalibPars();
	closeCalibFile();

	fillCalibParsV1();
       }
     printCalibPars();
}

//----------------

void CSPadCalibPars::openCalibFile ()
{
   string fname (m_calibdir);
   fname += "/"; 
   fname += m_cur_calibname; 
   fname += "/0-end.data"; // !!! THIS IS A SIMPLIFIED CASE OF THE FILE NAME!!!
   cout << "\nCSPadCalibPars::openCalibFile\n" << fname << endl;

   m_file.open(fname.c_str());
}

//----------------

void CSPadCalibPars::closeCalibFile ()
{
   m_file.close();
}

//----------------

void CSPadCalibPars::readCalibPars ()
{
  v_parameters.clear();
  std::string str;
  do{ 
      m_file >> str; 
      if(m_file.good()) {
         v_parameters.push_back(std::atof(str.c_str())); // cout << str << " "; 
      }
    } while( m_file.good() );                            // cout << endl << endl;
}

//----------------

void CSPadCalibPars::fillCalibParsV1 ()
{
       if( m_cur_calibname == v_calibname[0] ) m_center         = new CalibParsCenterV1(v_parameters);
  else if( m_cur_calibname == v_calibname[1] ) m_center_corr    = new CalibParsCenterCorrV1(v_parameters);
  else if( m_cur_calibname == v_calibname[2] ) m_marg_gap_shift = new CalibParsMargGapShiftV1(v_parameters);
  else if( m_cur_calibname == v_calibname[3] ) m_offset         = new CalibParsOffsetV1(v_parameters);
  else if( m_cur_calibname == v_calibname[4] ) m_offset_corr    = new CalibParsOffsetCorrV1(v_parameters);
  else if( m_cur_calibname == v_calibname[5] ) m_rotation       = new CalibParsRotationV1(v_parameters);
  else if( m_cur_calibname == v_calibname[6] ) m_tilt           = new CalibParsTiltV1(v_parameters);
  else if( m_cur_calibname == v_calibname[7] ) m_quad_rotation  = new CalibParsQuadRotationV1(v_parameters);
  else if( m_cur_calibname == v_calibname[8] ) m_quad_tilt      = new CalibParsQuadTiltV1(v_parameters);
}

//----------------

void CSPadCalibPars::printCalibPars()
{
    WithMsgLog("CSPadCalibPars", info, str) {
      str << "CSPadCalibPars::printCSPadCalibPars()" ;
    }        

     m_center         -> print();
     m_center_corr    -> print();
     m_marg_gap_shift -> print();
     m_offset         -> print();
     m_offset_corr    -> print();
     m_rotation       -> print();
     m_tilt           -> print();
     m_quad_rotation  -> print();
     m_quad_tilt      -> print();
}

//--------------
// Destructor --
//--------------

CSPadCalibPars::~CSPadCalibPars ()
{
  //delete [] m_data; 
}

//----------------
//----------------
//----------------
//----------------

CalibParsCenterV1::CalibParsCenterV1( const std::vector<float> v_parameters )
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
    size_t arr_size = sizeof( float ) * v_parameters.size()/3;
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
//----------------

CalibParsCenterCorrV1::CalibParsCenterCorrV1( const std::vector<float> v_parameters )
{
    if (v_parameters.size() != NUMBER_OF_PARAMETERS) {
        WithMsgLog("CalibParsCenterCorrV1", error, str) {
        str << "Expected number of parameters is " << NUMBER_OF_PARAMETERS ;
        str << ", read from file " << v_parameters.size() ;
        str << ": check the file.\n" ;
        }       
        abort();
    }
    size_t NPars    = NQuad * NSect;
    size_t arr_size = sizeof( float ) * v_parameters.size()/3;
    memcpy( &m_center_corr_x, &v_parameters[0],       arr_size );
    memcpy( &m_center_corr_y, &v_parameters[NPars],   arr_size );
    memcpy( &m_center_corr_z, &v_parameters[NPars*2], arr_size );
    //this->print();
}

void CalibParsCenterCorrV1::print()
{
  cout << endl << "X:" << endl;  
  for( int quad=0; quad<NQuad; ++quad ) {
    for( int sect=0; sect<NSect; ++sect ) { cout << "  " << m_center_corr_x[quad][sect]; }
    cout << endl;
  }
  cout << "Y:" << endl;  
  for( int quad=0; quad<NQuad; ++quad ) {
    for( int sect=0; sect<NSect; ++sect ) { cout << "  " << m_center_corr_y[quad][sect]; }
    cout << endl;
  }
  cout << "Z:" << endl;  
  for( int quad=0; quad<NQuad; ++quad ) {
    for( int sect=0; sect<NSect; ++sect ) { cout << "  " << m_center_corr_z[quad][sect]; }
    cout << endl;
  }
}

//----------------
//----------------

CalibParsMargGapShiftV1::CalibParsMargGapShiftV1( const std::vector<float> v_parameters )
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
//----------------

CalibParsOffsetV1::CalibParsOffsetV1( const std::vector<float> v_parameters )
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
    size_t arr_size = sizeof( float ) * v_parameters.size()/3;
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
//----------------

CalibParsOffsetCorrV1::CalibParsOffsetCorrV1( const std::vector<float> v_parameters )
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
    size_t arr_size = sizeof( float ) * v_parameters.size()/3;
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
//----------------

CalibParsTiltV1::CalibParsTiltV1( const std::vector<float> v_parameters )
{
    if (v_parameters.size() != NUMBER_OF_PARAMETERS) {
        WithMsgLog("CalibParsTiltV1", error, str) {
        str << "Expected number of parameters is " << NUMBER_OF_PARAMETERS ;
        str << ", read from file " << v_parameters.size() ;
        str << ": check the file.\n" ;
        }       
        abort();
    }
    size_t arr_size = sizeof( float ) * v_parameters.size();
    memcpy( &m_tilt, &v_parameters[0], arr_size );
    //this->print();
}

void CalibParsTiltV1::print()
{
  cout << endl << "Tilt:" << endl;  
  for( int quad=0; quad<NQuad; ++quad ) {
    for( int sect=0; sect<NSect; ++sect ) { cout << "  " << m_tilt[quad][sect]; }
    cout << endl;
  }
}

//----------------
//----------------

CalibParsQuadRotationV1::CalibParsQuadRotationV1( const std::vector<float> v_parameters )
{
    if (v_parameters.size() != NUMBER_OF_PARAMETERS) {
        WithMsgLog("CalibParsQuadRotationV1", error, str) {
        str << "Expected number of parameters is " << NUMBER_OF_PARAMETERS ;
        str << ", read from file " << v_parameters.size() ;
        str << ": check the file.\n" ;
        }       
        abort();
    }
    size_t arr_size = sizeof( float ) * v_parameters.size();
    memcpy( &m_quad_rotation, &v_parameters[0], arr_size );
    //this->print();
}

void CalibParsQuadRotationV1::print()
{
  cout << endl << "QuadRotation:" << endl;  
  for( int quad=0; quad<NQuad; ++quad ) { cout << "  " << m_quad_rotation[quad]; }
  cout << endl;
}

//----------------
//----------------

CalibParsQuadTiltV1::CalibParsQuadTiltV1( const std::vector<float> v_parameters )
{
    if (v_parameters.size() != NUMBER_OF_PARAMETERS) {
        WithMsgLog("CalibParsQuadTiltV1", error, str) {
        str << "Expected number of parameters is " << NUMBER_OF_PARAMETERS ;
        str << ", read from file " << v_parameters.size() ;
        str << ": check the file.\n" ;
        }       
        abort();
    }
    size_t arr_size = sizeof( float ) * v_parameters.size();
    memcpy( &m_quad_tilt, &v_parameters[0], arr_size );
    //this->print();
}

void CalibParsQuadTiltV1::print()
{
  cout << endl << "QuadTilt:" << endl;  
  for( int quad=0; quad<NQuad; ++quad ) { cout << "  " << m_quad_tilt[quad]; }
  cout << endl;
}

//----------------
//----------------


} // namespace CSPadImage
