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
       if( m_cur_calibname == v_calibname[0] ) m_center         = new pdscalibdata::CalibParsCenterV1(v_parameters);
  else if( m_cur_calibname == v_calibname[1] ) m_center_corr    = new pdscalibdata::CalibParsCenterCorrV1(v_parameters);
  else if( m_cur_calibname == v_calibname[2] ) m_marg_gap_shift = new pdscalibdata::CalibParsMargGapShiftV1(v_parameters);
  else if( m_cur_calibname == v_calibname[3] ) m_offset         = new pdscalibdata::CalibParsOffsetV1(v_parameters);
  else if( m_cur_calibname == v_calibname[4] ) m_offset_corr    = new pdscalibdata::CalibParsOffsetCorrV1(v_parameters);
  else if( m_cur_calibname == v_calibname[5] ) m_rotation       = new pdscalibdata::CalibParsRotationV1(v_parameters);
  else if( m_cur_calibname == v_calibname[6] ) m_tilt           = new pdscalibdata::CalibParsTiltV1(v_parameters);
  else if( m_cur_calibname == v_calibname[7] ) m_quad_rotation  = new pdscalibdata::CalibParsQuadRotationV1(v_parameters);
  else if( m_cur_calibname == v_calibname[8] ) m_quad_tilt      = new pdscalibdata::CalibParsQuadTiltV1(v_parameters);
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

} // namespace CSPadImage
