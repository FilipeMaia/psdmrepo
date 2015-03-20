//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadCalibIntensity...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSCalib/CSPadCalibIntensity.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <iomanip>  // for std::setw

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

const char logger[] = "PSCalib";

//----------------
// Constructors --
//----------------

  CSPadCalibIntensity::CSPadCalibIntensity(bool isTestMode)
  : PSCalib::CalibPars()
  , m_calibDir     (std::string())
  , m_typeGroupName(std::string())
  , m_source       (std::string())
  , m_runNumber    (0)
  , m_print_bits   (255)
{
    // Test staff:
    m_isTestMode = isTestMode;
    if (m_isTestMode) {
        m_calibdir      = "/reg/d/psdm/AMO/amotut13/calib/CsPad::CalibV1/Camp.0:Cspad.1";
        m_calibfilename = "1-end.data";
    }

    fillCalibNameVector ();
    loadCalibPars ();
    //printCalibPars();
}

//----------------

CSPadCalibIntensity::CSPadCalibIntensity ( const std::string&   calibDir,      //  "/reg/d/psdm/AMO/amoa1214/calib"
                                           const std::string&   typeGroupName, //  "CsPad::CalibV1"
                                           const std::string&   source,        //  "Camp.0:Cspad.0"
                                           const unsigned long& runNumber,     //  10
                                           unsigned             print_bits)
  : PSCalib::CalibPars()
  , m_calibDir(calibDir)
  , m_typeGroupName(typeGroupName)
  , m_source(source)
  , m_runNumber(runNumber)
  , m_print_bits(print_bits)
{
    m_isTestMode = false;

    fillCalibNameVector ();
    loadCalibPars ();

    if( m_print_bits & 1 ) printInputPars ();
    //printCalibPars();
}

//----------------

CSPadCalibIntensity::CSPadCalibIntensity ( const std::string&   calibDir,      //  "/reg/d/psdm/AMO/amoa1214/calib"
                                           const std::string&   typeGroupName, //  "CsPad::CalibV1"
                                           const Pds::Src&      src,           //  Pds::Src m_src; <- is defined in get(...,&m_src)
                                           const unsigned long& runNumber,     //  10
                                           unsigned             print_bits)
  : PSCalib::CalibPars()
  , m_calibDir(calibDir)
  , m_typeGroupName(typeGroupName)
  , m_source(std::string()) // "in this constructor source is defined through Pds::Src"
  , m_src(src)
  , m_runNumber(runNumber)
  , m_print_bits(print_bits)
{
    m_isTestMode = false;

    fillCalibNameVector ();
    loadCalibPars ();

    if( m_print_bits & 1 ) printInputPars ();
    //printCalibPars();
}

//----------------

void CSPadCalibIntensity::fillCalibNameVector ()
{
    v_calibname.clear();
    v_calibname.push_back("pedestals");
    v_calibname.push_back("pixel_status");
    v_calibname.push_back("common_mode");
    v_calibname.push_back("pixel_gain");
    v_calibname.push_back("pixel_rms");
}

//----------------

void CSPadCalibIntensity::loadCalibPars ()
{
    for( vector<std::string>::const_iterator iterCalibName  = v_calibname.begin();
                                             iterCalibName != v_calibname.end(); iterCalibName++ )
      {
        m_cur_calibname = *iterCalibName;

	getCalibFileName();

        if (m_fname == std::string()) { 
	  fillDefaultCalibParsV1 ();
          if ( m_print_bits & 4 ) msgUseDefault ();
          m_calibtype_status[m_cur_calibname] = 0; 
        } 
        else 
        {
	  fillCalibParsV1 ();
          m_calibtype_status[m_cur_calibname] = 1; 
	}
      }
}

//----------------

void CSPadCalibIntensity::getCalibFileName ()
{
  if ( m_isTestMode ) 
    {
      m_fname  = m_calibdir + "/"; 
      m_fname += m_cur_calibname + "/"; 
      m_fname += m_calibfilename; // "/0-end.data"; // !!! THIS IS A SIMPLIFIED CASE OF THE FILE NAME!!!
    }
  else if (m_calibDir == std::string())
    {
      m_fname = std::string();
    }
  else
    {
      unsigned print_bits_cff = ( m_print_bits & 2 ) ? 255 : 0;

      PSCalib::CalibFileFinder *calibfinder = new PSCalib::CalibFileFinder(m_calibDir, m_typeGroupName, print_bits_cff);
      //m_fname = calibfinder -> findCalibFile(m_src, m_cur_calibname, m_runNumber);

      if (m_source == std::string())
          m_fname = calibfinder -> findCalibFile(m_src, m_cur_calibname, m_runNumber);
      else
          m_fname = calibfinder -> findCalibFile(m_source, m_cur_calibname, m_runNumber);
    }
  MsgLog("CSPadCalibIntensity", debug, "getCalibFileName(): " << m_fname);
}

//----------------

void CSPadCalibIntensity::fillCalibParsV1 ()
{
  if     ( m_cur_calibname == v_calibname[0] ) m_pedestals    = new pdscalibdata::CsPadPedestalsV2  (m_fname);
  else if( m_cur_calibname == v_calibname[1] ) m_pixel_status = new pdscalibdata::CsPadPixelStatusV2(m_fname);
  else if( m_cur_calibname == v_calibname[2] ) m_common_mode  = new pdscalibdata::CsPadCommonModeV2 (m_fname);
  else if( m_cur_calibname == v_calibname[3] ) m_pixel_gain   = new pdscalibdata::CsPadPixelGainV2  (m_fname);
  else if( m_cur_calibname == v_calibname[4] ) m_pixel_rms    = new pdscalibdata::CsPadPixelRmsV2   (m_fname);
}

//----------------

void CSPadCalibIntensity::fillDefaultCalibParsV1 ()
{
  // If default parameters are available - set them.
  // For calib types where default parameters are not accaptable and the file is missing - error message and abort.
  if     ( m_cur_calibname == v_calibname[0] ) m_pedestals    = new pdscalibdata::CsPadPedestalsV2  ();
  else if( m_cur_calibname == v_calibname[1] ) m_pixel_status = new pdscalibdata::CsPadPixelStatusV2();
  else if( m_cur_calibname == v_calibname[2] ) m_common_mode  = new pdscalibdata::CsPadCommonModeV2 ();
  else if( m_cur_calibname == v_calibname[3] ) m_pixel_gain   = new pdscalibdata::CsPadPixelGainV2  ();
  else if( m_cur_calibname == v_calibname[4] ) m_pixel_rms    = new pdscalibdata::CsPadPixelRmsV2   ();

  else if( m_print_bits & 8 ) fatalMissingFileName ();
}

//----------------

void CSPadCalibIntensity::fatalMissingFileName ()
{
	MsgLog("CSPadCalibIntensity", warning, "In fillDefaultCalibParsV1(): the calibration file for the source=" << m_source 
                  << ", type=" << m_cur_calibname 
                  << ", run=" <<  m_runNumber
                  << " is not found ..."
	          << "\nWARNING: Default CSPAD intensity correction constants can not guarantee correct intensity transformation..."
	          << "\nWARNING: Please provide all expected CSPAD intensity correction constants under the directory .../<experiment>/calib/...");
	abort();
}

//----------------

void CSPadCalibIntensity::msgUseDefault ()
{
	MsgLog("CSPadCalibIntensity", info, "In getCalibFileName(): the calibration file for the source=" << m_source 
                  << ", type=" << m_cur_calibname 
                  << ", run=" <<  m_runNumber
                  << " is not found ..."
	          << "\nWARNING: Default CSPAD alignment constants will be used.");
}

//----------------

void CSPadCalibIntensity::printCalibPars()
{
    WithMsgLog("CSPadCalibIntensity", info, str) {
      str << "printCalibPars()" ;
      //str << "\n getColSize_um()    = " << getColSize_um() ;
      //str << "\n getRowSize_um()    = " << getRowSize_um() ;
    }        

    m_pedestals    -> print();
    m_pixel_status -> print();
    m_common_mode  -> print();
    m_pixel_gain   -> print();
    m_pixel_rms    -> print();
}

//----------------

void CSPadCalibIntensity::printInputPars()
{
    WithMsgLog("CSPadCalibIntensity", info, str) {
      str << "printInputPars()" ;
      str << "\n m_calibDir      = " << m_calibDir ;
      str << "\n m_typeGroupName = " << m_typeGroupName ;
      str << "\n m_source        = " << m_source ;
      str << "\n m_runNumber     = " << m_runNumber ;
      str << "\n m_print_bits    = " << m_print_bits ;
    }        
}

//----------------

void CSPadCalibIntensity::printCalibParsStatus ()
{
    WithMsgLog("CSPadCalibIntensity", info, str) {
      str << "printCalibParsStatus()" ;

      for( vector<std::string>::const_iterator iterCalibName  = v_calibname.begin();
                                               iterCalibName != v_calibname.end(); iterCalibName++ )
      {
          m_cur_calibname = *iterCalibName;
          str << "\n type: "  << std::left << std::setw(20) << m_cur_calibname
              << " status = " << m_calibtype_status[m_cur_calibname]; 
      }
    }        
}

//--------------
// Destructor --
//--------------

CSPadCalibIntensity::~CSPadCalibIntensity ()
{
  //delete [] m_data; 
}

//----------------
//----------------
//----------------
//----------------

} // namespace PSCalib
