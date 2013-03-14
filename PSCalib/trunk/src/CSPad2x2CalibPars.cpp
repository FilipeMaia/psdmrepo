//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPad2x2CalibPars...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSCalib/CSPad2x2CalibPars.h"

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

CSPad2x2CalibPars::CSPad2x2CalibPars ()
{
  cout << "CSPad2x2CalibPars::CSPad2x2CalibPars" 
       << "\nHere we have to find from the xtc_file_name the run number and find the calib directory..." << endl;

    // Temporary staff:

    m_isTestMode = true;

    //m_calibdir      = "/reg/neh/home1/dubrovin/LCLS/CSPad2x2Alignment/calib-cspad2x2-01-2013-02-13/";
    m_calibdir      = "/reg/d/psdm/mec/mec73313/calib";
    m_calibfilename = "0-end.data";

    fillCalibNameVector ();
    loadCalibPars ();
    //printCalibPars();
}

//----------------


CSPad2x2CalibPars::CSPad2x2CalibPars ( const std::string&   calibDir,      //  /reg/d/psdm/cxi/cxi35711/calib
                                       const std::string&   typeGroupName, //  CsPad::CalibV1
                                       const std::string&   source,        //  CxiDs1.0:Cspad.0
                                       const unsigned long& runNumber )    //  10
  : m_calibDir(calibDir)
  , m_typeGroupName(typeGroupName)
  , m_source(source)
  , m_runNumber(runNumber)
{
    m_isTestMode = false;

    fillCalibNameVector ();
    loadCalibPars ();

    printInputPars ();
    //printCalibPars();
}


//----------------

CSPad2x2CalibPars::CSPad2x2CalibPars ( const std::string&   calibDir,      //  /reg/d/psdm/cxi/cxi35711/calib
                                       const std::string&   typeGroupName, //  CsPad::CalibV1
                                       const Pds::Src&      src,           //  Pds::Src m_src; <- is defined in get(...,&m_src)
                                       const unsigned long& runNumber )    //  10
  : m_calibDir(calibDir)
  , m_typeGroupName(typeGroupName)
  , m_source(std::string())
  , m_src(src)
  , m_runNumber(runNumber)
{
    m_isTestMode = false;

    fillCalibNameVector ();
    loadCalibPars ();

    printInputPars ();
    //printCalibPars();
}

//----------------

void CSPad2x2CalibPars::fillCalibNameVector ()
{
    v_calibname.clear();
    v_calibname.push_back("center");
    v_calibname.push_back("tilt");
}

//----------------

void CSPad2x2CalibPars::loadCalibPars ()
{
    for( vector<std::string>::const_iterator iterCalibName  = v_calibname.begin();
                                             iterCalibName != v_calibname.end(); iterCalibName++ )
      {
        m_cur_calibname = *iterCalibName;

	getCalibFileName();

        if (m_fname == std::string()) { 
	  fillDefaultCalibParsV1 ();
          msgUseDefault ();
        } 
        else 
        {
	  openCalibFile   ();
	  readCalibPars   ();
	  closeCalibFile  ();
	  fillCalibParsV1 ();
	}
      }
}

//----------------

void CSPad2x2CalibPars::getCalibFileName ()
{
  if ( m_isTestMode ) 
    {
      m_fname  = m_calibdir + "/"; 
      m_fname += m_cur_calibname + "/"; 
      m_fname += m_calibfilename; // "/0-end.data"; // !!! THIS IS A SIMPLIFIED CASE OF THE FILE NAME!!!
    }
  else
    {
      PSCalib::CalibFileFinder *calibfinder = new PSCalib::CalibFileFinder(m_calibDir, m_typeGroupName);
      //m_fname = calibfinder -> findCalibFile(m_src, m_cur_calibname, m_runNumber);

      if (m_source == std::string())
          m_fname = calibfinder -> findCalibFile(m_src, m_cur_calibname, m_runNumber);
      else
          m_fname = calibfinder -> findCalibFile(m_source, m_cur_calibname, m_runNumber);
    }
  MsgLog("CSPad2x2CalibPars", debug, "getCalibFileName(): " << m_fname);
}

//----------------

void CSPad2x2CalibPars::openCalibFile ()
{
   m_file.open(m_fname.c_str());

   if (not m_file.good()) {
     const std::string msg = "Failed to open file: "+m_fname;
     MsgLogRoot(error, msg);
     //throw std::runtime_error(msg);
   }
}

//----------------

void CSPad2x2CalibPars::closeCalibFile ()
{
   m_file.close();
}

//----------------

void CSPad2x2CalibPars::readCalibPars ()
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

void CSPad2x2CalibPars::fillCalibParsV1 ()
{
       if( m_cur_calibname == v_calibname[0] ) m_center = new pdscalibdata::CsPad2x2CenterV1(v_parameters);
  else if( m_cur_calibname == v_calibname[1] ) m_tilt   = new pdscalibdata::CsPad2x2TiltV1(v_parameters);
}

//----------------

void CSPad2x2CalibPars::fillDefaultCalibParsV1 ()
{
  // If default parameters are available - set them.
  // For calib types where default parameters are not accaptable and the file is missing - error message and abort.
       if( m_cur_calibname == v_calibname[0] ) m_center = new pdscalibdata::CsPad2x2CenterV1();
  else if( m_cur_calibname == v_calibname[1] ) m_tilt   = new pdscalibdata::CsPad2x2TiltV1();

  else fatalMissingFileName ();
}

//----------------

void CSPad2x2CalibPars::fatalMissingFileName ()
{
	MsgLog("CSPad2x2CalibPars", warning, "In getCalibFileName(): the calibration file for the source=" << m_source 
                  << ", type=" << m_cur_calibname 
                  << ", run=" <<  m_runNumber
                  << " is not found ..."
	          << "\nWARNING: Default CSPad2x2 alignment constants can not guarantee correct geometry and are not available yet."
	          << "\nWARNING: Please provide all expected CSPad alignment constants under the directory .../<experiment>/calib/...");
	abort();
}

//----------------

void CSPad2x2CalibPars::msgUseDefault ()
{
	MsgLog("CSPad2x2CalibPars", warning, "In getCalibFileName(): the calibration file for the source=" << m_source 
                  << ", type=" << m_cur_calibname 
                  << ", run=" <<  m_runNumber
                  << " is not found ..."
	          << "\nWARNING: Default CSPad2x2 alignment constants will be used.");
}

//----------------

void CSPad2x2CalibPars::printCalibPars()
{
    WithMsgLog("CSPad2x2CalibPars", info, str) {
      str << "printCalibPars()" ;
      str << "\n getColSize_um()    = " << getColSize_um() ;
      str << "\n getRowSize_um()    = " << getRowSize_um() ;
      str << "\n getGapRowSize_um() = " << getGapRowSize_um() ;
    }        

     m_center         -> print();
     m_tilt           -> print();
}

//----------------

void CSPad2x2CalibPars::printInputPars()
{
    WithMsgLog("CSPad2x2CalibPars", info, str) {
      str << "printInputPars()" ;
      str << "\n m_calibDir      = " << m_calibDir ;
      str << "\n m_typeGroupName = " << m_typeGroupName ;
      str << "\n m_source        = " << m_source ;
      str << "\n m_runNumber     = " << m_runNumber ;
    }        
}

//--------------
// Destructor --
//--------------

CSPad2x2CalibPars::~CSPad2x2CalibPars ()
{
  //delete [] m_data; 
}

//----------------
//----------------
//----------------
//----------------

} // namespace PSCalib
