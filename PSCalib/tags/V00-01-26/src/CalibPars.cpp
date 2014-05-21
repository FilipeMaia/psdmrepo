//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibPars...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSCalib/CalibPars.h"

//-----------------
// C/C++ Headers --
//-----------------
//#include <iostream>
//#include <iomanip>  // for std::setw

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
using namespace std;

namespace PSCalib {

//----------------------------------------
//-- Public Function Member Definitions --
//----------------------------------------

//-----------------------------
const size_t 
CalibPars::ndim()
{
  default_msg("ndim()");
  return 0;
}

//-----------------------------
const size_t 
CalibPars::size()
{
  default_msg("size()");
  return 0;
}

//-----------------------------
const unsigned* 
CalibPars::shape() 
{
  default_msg("shape()");
  return 0;
}

//-----------------------------
const CalibPars::pedestals_t* 
CalibPars::pedestals()
{ 
  default_msg("pedestals()");
  return 0;
}

//-----------------------------
const CalibPars::pixel_status_t* 
CalibPars::pixel_status()
{ 
  default_msg("pixel_status()");
  return 0;
}

//-----------------------------
const CalibPars::pixel_gain_t* 
CalibPars::pixel_gain()
{ 
  default_msg("pixel_gain()");
  return 0;
}

//-----------------------------
const CalibPars::pixel_rms_t* 
CalibPars::pixel_rms()
{ 
  default_msg("pixel_rms()");
  return 0;
}

//-----------------------------
const CalibPars::common_mode_t* 
CalibPars::common_mode()
{ 
  default_msg("common_mode()");
  return 0;
}

//-----------------------------
void 
CalibPars::printCalibPars()
{
  default_msg("printCalibPars()");
}

//-----------------------------
void 
CalibPars::default_msg(const std::string& msg) 
{
  MsgLog("PSCalib", warning, "DEFAULT METHOD "<< msg << " SHOULD BE RE-IMPLEMENTED IN DERIVED CLASS.");
}

//-----------------------------
void 
CalibPars::fill_map_type2str() 
{
  map_type2str[PEDESTALS]    = std::string("pedestals");
  map_type2str[PIXEL_STATUS] = std::string("pixel_status");
  map_type2str[PIXEL_RMS]    = std::string("pixel_rms");
  map_type2str[PIXEL_GAIN]   = std::string("pixel_gain");
  map_type2str[COMMON_MODE]  = std::string("common_mode");
}


//-----------------------------
void 
CalibPars::printCalibTypes() 
{
    WithMsgLog("PSCalib", info, str) {
      str << "print_calib_types():\n  Map map_type2str of known enumerated data types:\n" ;
      for (std::map<CALIB_TYPE, std::string>::iterator it=map_type2str.begin(); it!=map_type2str.end(); ++it)
          str << "    " << it->first << " : " << it->second << '\n';
    }
}

//-----------------------------
} // namespace PSCalib
//-----------------------------
