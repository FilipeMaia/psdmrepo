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
#include "PSCalib/CSPadCalibPars.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSCalib/CalibFileFinder.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSCalib {

//----------------
// Constructors --
//----------------

CSPadCalibPars::CSPadCalibPars ()
{
  cout << "CSPadCalibPars::CSPadCalibPars" 
       << "\nHere we have to find from the xtc_file_name the run number and find the calib directory..." << endl;

    // Temporary staff:

    m_isTestMode = true;

    m_calibdir      = "/reg/neh/home/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi35711-r0009-det";
    m_calibfilename = "0-end.data";

    fillCalibNameVector ();
    loadCalibPars ();
    //printCalibPars();
}

//----------------

CSPadCalibPars::CSPadCalibPars ( const std::string&   calibDir,           //  /reg/d/psdm/cxi/cxi35711/calib
                                 const std::string&   typeGroupName,      //  CsPad::CalibV1
                                 const std::string&   source,             //  CxiDs1.0:Cspad.0
                                 const unsigned long& runNumber )         //  10
  : m_calibDir(calibDir)
  , m_typeGroupName(typeGroupName)
  , m_source(source)
  , m_runNumber(runNumber)
{
    m_isTestMode = false;

    fillCalibNameVector ();
    loadCalibPars ();
    //printCalibPars();
}

//----------------

void CSPadCalibPars::fillCalibNameVector ()
{
    v_calibname.clear();
    v_calibname.push_back("center");
    v_calibname.push_back("center_corr");
    v_calibname.push_back("marg_gap_shift");
    v_calibname.push_back("offset");
    v_calibname.push_back("offset_corr");
    v_calibname.push_back("rotation");
    v_calibname.push_back("tilt");
    v_calibname.push_back("quad_rotation");
    v_calibname.push_back("quad_tilt");
}

//----------------

void CSPadCalibPars::loadCalibPars ()
{
    for( vector<std::string>::const_iterator iterCalibName  = v_calibname.begin();
                                             iterCalibName != v_calibname.end(); iterCalibName++ )
      {
        m_cur_calibname = *iterCalibName;

	getCalibFileName();
	openCalibFile   ();
	readCalibPars   ();
	closeCalibFile  ();
	fillCalibParsV1 ();
      }
}

//----------------

void CSPadCalibPars::getCalibFileName ()
{
  if ( m_isTestMode ) 
    {
      m_fname  = m_calibdir; 
      m_fname += "/"; 
      m_fname += m_cur_calibname; 
      m_fname += "/"; 
      m_fname += m_calibfilename; // "/0-end.data"; // !!! THIS IS A SIMPLIFIED CASE OF THE FILE NAME!!!
    }
  else
    {
      PSCalib::CalibFileFinder *calibfinder = new PSCalib::CalibFileFinder(m_calibDir, m_typeGroupName);
      m_fname = calibfinder -> findCalibFile(m_source, m_cur_calibname, m_runNumber);

      // Check if the file name is empty:
      if (m_fname == std::string()) { 
	MsgLog("CSPadCalibPars", warning, "In getCalibFileName(): the calibration file for the source=" << m_source 
                  << ", type=" << m_cur_calibname 
                  << ", run=" <<  m_runNumber
                  << " is not found ..."
	          << "\nWARNING: Default CSPad alignment constants can not guarantee correct geometry and are not available yet."
	          << "\nWARNING: Please provide all expected CSPad alignment constants under the directory .../<experiment>/calib/...");
	abort();
      }
    }
  MsgLog("CSPadCalibPars", debug, "getCalibFileName(): " << m_fname);
}

//----------------

void CSPadCalibPars::openCalibFile ()
{
   m_file.open(m_fname.c_str());

   if (not m_file.good()) {
     const std::string msg = "Failed to open file: "+m_fname;
     MsgLogRoot(error, msg);
     //throw std::runtime_error(msg);
   }
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
      str << "printCSPadCalibPars()" ;
      str << "\n getColSize_um()    = " << getColSize_um() ;
      str << "\n getRowSize_um()    = " << getRowSize_um() ;
      str << "\n getGapRowSize_um() = " << getGapRowSize_um() ;
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

} // namespace PSCalib
