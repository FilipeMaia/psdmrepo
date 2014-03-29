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

CSPadCalibPars::CSPadCalibPars (bool isTestMode)
  : m_calibDir     (std::string())
  , m_typeGroupName(std::string())
  , m_source       (std::string())
  , m_runNumber    (0)
{
    // Test staff:
    m_isTestMode = isTestMode;
    if (m_isTestMode) {
        m_calibdir      = "/reg/neh/home/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi35711-r0009-det";
        m_calibfilename = "0-end.data";
    }

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
//, m_src(source)
  , m_runNumber(runNumber)
{
    m_isTestMode = false;

    fillCalibNameVector ();
    loadCalibPars ();

    MsgLog("CSPadCalibPars", info, "Depricated constructor with string& source");
    printInputPars ();
    //printCalibPars();
}

//----------------

CSPadCalibPars::CSPadCalibPars ( const std::string&   calibDir,           //  /reg/d/psdm/cxi/cxi35711/calib
                                 const std::string&   typeGroupName,      //  CsPad::CalibV1
                                 const Pds::Src&      src,                //  Pds::Src m_src; <- is defined in get(...,&m_src)
                                 const unsigned long& runNumber )         //  10
  : m_calibDir(calibDir)
  , m_typeGroupName(typeGroupName)
  , m_source(std::string())
  , m_src(src)
  , m_runNumber(runNumber)
{
    m_isTestMode = false;

    fillCalibNameVector ();
    loadCalibPars ();

    MsgLog("CSPadCalibPars", info, "Regular constructor with Pds::Src& src, hence m_source is empty...");
    printInputPars ();
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
    v_calibname.push_back("beam_vector");
    v_calibname.push_back("beam_intersect");
    v_calibname.push_back("center_global");
    v_calibname.push_back("rotation_global");
}

//----------------

void CSPadCalibPars::loadCalibPars ()
{
    for( vector<std::string>::const_iterator iterCalibName  = v_calibname.begin();
                                             iterCalibName != v_calibname.end(); iterCalibName++ )
      {
        m_cur_calibname = *iterCalibName;

	getCalibFileName();

        if (m_fname == std::string()) { 
	  fillDefaultCalibParsV1 ();
          msgUseDefault ();
          m_calibtype_status[m_cur_calibname] = 0; 
        } 
        else 
        {
	  openCalibFile   ();
	  readCalibPars   ();
	  closeCalibFile  ();
	  fillCalibParsV1 ();
          m_calibtype_status[m_cur_calibname] = 1; 
	}
      }
}

//----------------

void CSPadCalibPars::getCalibFileName ()
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
      PSCalib::CalibFileFinder *calibfinder = new PSCalib::CalibFileFinder(m_calibDir, m_typeGroupName);
      //m_fname = calibfinder -> findCalibFile(m_src, m_cur_calibname, m_runNumber);

      if (m_source == std::string())
          m_fname = calibfinder -> findCalibFile(m_src, m_cur_calibname, m_runNumber);
      else
          m_fname = calibfinder -> findCalibFile(m_source, m_cur_calibname, m_runNumber);
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
  else if( m_cur_calibname == v_calibname[9] ) m_beam_vector    = new pdscalibdata::CsPadBeamVectorV1(v_parameters);
  else if( m_cur_calibname == v_calibname[10]) m_beam_intersect = new pdscalibdata::CsPadBeamIntersectV1(v_parameters);
  else if( m_cur_calibname == v_calibname[11]) m_center_global  = new pdscalibdata::CsPadCenterGlobalV1(v_parameters);
  else if( m_cur_calibname == v_calibname[12]) m_rotation_global= new pdscalibdata::CsPadRotationGlobalV1(v_parameters);
}

//----------------

void CSPadCalibPars::fillDefaultCalibParsV1 ()
{
  // If default parameters are available - set them.
  // For calib types where default parameters are not accaptable and the file is missing - error message and abort.
       if( m_cur_calibname == v_calibname[0] ) m_center         = new pdscalibdata::CalibParsCenterV1();
  else if( m_cur_calibname == v_calibname[1] ) m_center_corr    = new pdscalibdata::CalibParsCenterCorrV1();
  else if( m_cur_calibname == v_calibname[2] ) m_marg_gap_shift = new pdscalibdata::CalibParsMargGapShiftV1();
  else if( m_cur_calibname == v_calibname[3] ) m_offset         = new pdscalibdata::CalibParsOffsetV1();
  else if( m_cur_calibname == v_calibname[4] ) m_offset_corr    = new pdscalibdata::CalibParsOffsetCorrV1();
  else if( m_cur_calibname == v_calibname[5] ) m_rotation       = new pdscalibdata::CalibParsRotationV1();
  else if( m_cur_calibname == v_calibname[6] ) m_tilt           = new pdscalibdata::CalibParsTiltV1();
  else if( m_cur_calibname == v_calibname[7] ) m_quad_rotation  = new pdscalibdata::CalibParsQuadRotationV1();
  else if( m_cur_calibname == v_calibname[8] ) m_quad_tilt      = new pdscalibdata::CalibParsQuadTiltV1();
  else if( m_cur_calibname == v_calibname[9] ) m_beam_vector    = new pdscalibdata::CsPadBeamVectorV1();
  else if( m_cur_calibname == v_calibname[10]) m_beam_intersect = new pdscalibdata::CsPadBeamIntersectV1();
  else if( m_cur_calibname == v_calibname[11]) m_center_global  = new pdscalibdata::CsPadCenterGlobalV1();
  else if( m_cur_calibname == v_calibname[12]) m_rotation_global= new pdscalibdata::CsPadRotationGlobalV1();

  else fatalMissingFileName ();
}

//----------------

void CSPadCalibPars::fatalMissingFileName ()
{
	MsgLog("CSPadCalibPars", warning, "In getCalibFileName(): the calibration file for the source=" << m_source 
                  << ", type=" << m_cur_calibname 
                  << ", run=" <<  m_runNumber
                  << " is not found ..."
	          << "\nWARNING: Default CSPad alignment constants can not guarantee correct geometry and are not available yet."
	          << "\nWARNING: Please provide all expected CSPad alignment constants under the directory .../<experiment>/calib/...");
	abort();
}

//----------------

void CSPadCalibPars::msgUseDefault ()
{
	MsgLog("CSPadCalibPars", info, "In getCalibFileName(): the calibration file for the source=" << m_source 
                  << ", type=" << m_cur_calibname 
                  << ", run=" <<  m_runNumber
                  << " is not found ..."
	          << "\nWARNING: Default CSPad alignment constants will be used.");
}

//----------------

void CSPadCalibPars::printCalibPars()
{
    WithMsgLog("CSPadCalibPars", info, str) {
      str << "printCalibPars()" ;
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
     m_beam_vector    -> print();
     m_beam_intersect -> print();
     m_center_global  -> print();
     m_rotation_global-> print();
}

//----------------

void CSPadCalibPars::printInputPars()
{
    WithMsgLog("CSPadCalibPars", info, str) {
      str << "printInputPars()" ;
      str << "\n m_calibDir      = " << m_calibDir ;
      str << "\n m_typeGroupName = " << m_typeGroupName ;
      str << "\n m_source        = " << m_source ;
      str << "\n m_runNumber     = " << m_runNumber ;
    }        
}

//----------------

void CSPadCalibPars::printCalibParsStatus ()
{
    WithMsgLog("CSPadCalibPars", info, str) {
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

CSPadCalibPars::~CSPadCalibPars ()
{
  //delete [] m_data; 
}

//----------------
//----------------
//----------------
//----------------

} // namespace PSCalib
