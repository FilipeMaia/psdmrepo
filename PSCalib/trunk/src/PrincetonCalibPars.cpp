//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonCalibPars...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSCalib/PrincetonCalibPars.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <iomanip>   // for std::setw
#include <sstream>   // for stringstream

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

PrincetonCalibPars::PrincetonCalibPars ( const std::string&   calibDir,  //  "/reg/d/psdm/AMO/amoa1214/calib"
                                         const std::string&   groupName, //  "PNCCD::CalibV1"
                                         const std::string&   source,    //  "Camp.0:pnCCD.0"
                                         const unsigned long& runNumber, //  10
                                         unsigned             print_bits)
  : PSCalib::CalibPars()
  , pdscalibdata::PrincetonBaseV1()
  , m_calibDir(calibDir)
  , m_groupName(groupName)
  , m_source(source)
  , m_runNumber(runNumber)
  , m_print_bits(print_bits)
{
    init();
}

//----------------

PrincetonCalibPars::PrincetonCalibPars ( const std::string&   calibDir,  //  "/reg/d/psdm/AMO/amoa1214/calib"
                                         const std::string&   groupName, //  "PNCCD::CalibV1"
                                         const Pds::Src&      src,       //  Pds::Src m_src; <- is defined in get(...,&m_src)
                                         const unsigned long& runNumber, //  10
                                         unsigned             print_bits)
  : PSCalib::CalibPars()
  , m_calibDir(calibDir)
  , m_groupName(groupName)
  , m_source(std::string()) // "in this constructor source is defined through Pds::Src"
  , m_src(src)
  , m_runNumber(runNumber)
  , m_print_bits(print_bits)
{
    init();
}

//----------------

void PrincetonCalibPars::init()
{
  m_pedestals    = 0;
  m_pixel_gain   = 0;
  m_pixel_rms    = 0;
  m_common_mode  = 0;
  m_pixel_status = 0; 
  m_name = std::string("PrincetonCalibPars");

  if( m_print_bits & 1 ) printInputPars();
  if( m_print_bits & 16) printCalibTypes(); // method from superclass
  m_prbits_cff  = ( m_print_bits &  2 ) ? 0377 : 0;
  m_prbits_type = ( m_print_bits & 32 ) ?    1 : 0;
}

//----------------

std::string PrincetonCalibPars::getCalibFileName (const CALIB_TYPE& calibtype)
{
  std::string fname = std::string();
  if (m_calibDir != std::string()) {

      PSCalib::CalibFileFinder *calibfinder = new PSCalib::CalibFileFinder(m_calibDir, m_groupName, m_prbits_cff);

      if (m_source == std::string())
          fname = calibfinder -> findCalibFile(m_src, map_type2str[calibtype], m_runNumber);
      else
          fname = calibfinder -> findCalibFile(m_source, map_type2str[calibtype], m_runNumber);
  }

  if( m_print_bits & 4 ) MsgLog(m_name, info, "Use calibration parameters from file: " << fname);
  if( m_print_bits & 8 && fname.empty() )
       MsgLog(m_name, info, "File for calibration type " << map_type2str[calibtype]
	      << " IS MISSING! Use default calibration parameters on your own risk..."); 

  return fname;
}

//----------------

const CalibPars::pedestals_t*
PrincetonCalibPars::pedestals()
{
  if (m_pedestals == 0) {
      std::string fname = getCalibFileName(PEDESTALS);
      m_pedestals = new NDAIPEDS(fname, shape_base(), pedestals_t(0), m_prbits_type);
  }
  return m_pedestals->get_ndarray().data();
}

//----------------

const CalibPars::pixel_status_t*
PrincetonCalibPars::pixel_status()
{
  if (m_pixel_status == 0) {
      std::string fname = getCalibFileName(PIXEL_STATUS);
      m_pixel_status = new NDAISTATUS(fname, shape_base(), pixel_status_t(1), m_prbits_type);
  }
  return m_pixel_status->get_ndarray().data();
}

//----------------

const CalibPars::pixel_gain_t*
PrincetonCalibPars::pixel_gain()
{
  if (m_pixel_gain == 0) {
      std::string fname = getCalibFileName(PIXEL_GAIN);
      m_pixel_gain = new NDAIGAIN(fname, shape_base(), pixel_gain_t(1), m_prbits_type);
  }
  return m_pixel_gain->get_ndarray().data();
}

//----------------

const CalibPars::pixel_rms_t*
PrincetonCalibPars::pixel_rms()
{
  if (m_pixel_rms == 0) {
      std::string fname = getCalibFileName(PIXEL_RMS);
      m_pixel_rms = new NDAIRMS(fname, shape_base(), pixel_rms_t(1), m_prbits_type);
  }
  return m_pixel_rms->get_ndarray().data();
}

//----------------

const CalibPars::common_mode_t*
PrincetonCalibPars::common_mode()
{
  if (m_common_mode == 0) {
      std::string fname = getCalibFileName(COMMON_MODE);
      ndarray<const CalibPars::common_mode_t,1> nda = make_ndarray(cmod_base(), SizeCM); // see PrincetonBaseV1
      m_common_mode = new NDAICMOD(fname, nda, m_prbits_type);
  }
  return m_common_mode->get_ndarray().data();
}

//----------------
//----------------
//----------------
//----------------

void PrincetonCalibPars::printInputPars()
{
    WithMsgLog(m_name, info, str) {
      str << "printInputPars()"
      	  << "\n m_calibDir   = " << m_calibDir 
      	  << "\n m_groupName  = " << m_groupName 
      	  << "\n m_source     = " << m_source 
      	  << "\n m_runNumber  = " << m_runNumber 
      	  << "\n m_print_bits = " << m_print_bits
      	  << "\nDetector configuration parameters:"
	  << "\n Ndim = " << Ndim
	  << "\n Rows = " << Rows
	  << "\n Cols = " << Cols
          << "\n Shape = [" << shape_base()[0]
          << ","            << shape_base()[1]
          << "]";
    }
      //std::string str = map_type2str[PEDESTALS];
      //std::cout << "map_type2str[PEDESTALS] = " << str << '\n';
}

//----------------

void PrincetonCalibPars::printCalibParsStatus ()
{
    std::stringstream smsg; 
    smsg << "\n  printCalibParsStatus() for:"
    	 << "\n  pedestals    : " << m_pedestals    -> str_status()
    	 << "\n  pixel_status : " << m_pixel_status -> str_status()
    	 << "\n  pixel_gain   : " << m_pixel_gain   -> str_status()
    	 << "\n  pixel_rms    : " << m_pixel_rms    -> str_status()
    	 << "\n  common_mode  : " << m_common_mode  -> str_status();
    MsgLog(m_name, info, smsg.str());
}

//----------------

void PrincetonCalibPars::printCalibPars()
{
    std::stringstream smsg; 
    smsg << "printCalibPars()"
         << "\n  shape = ["       << m_pedestals    -> str_shape() << "]"
    	 << "\n  pedestals    : " << m_pedestals    -> str_ndarray_info()
    	 << "\n  pixel_status : " << m_pixel_status -> str_ndarray_info()
    	 << "\n  pixel_gain   : " << m_pixel_gain   -> str_ndarray_info()
    	 << "\n  pixel_rms    : " << m_pixel_rms    -> str_ndarray_info()
    	 << "\n  common_mode  : " << m_common_mode  -> str_ndarray_info();
    MsgLog(m_name, info, smsg.str());
}

//--------------
// Destructor --
//--------------

PrincetonCalibPars::~PrincetonCalibPars ()
{
  //delete [] m_data; 
}

//----------------
//----------------
//----------------
//----------------

} // namespace PSCalib
