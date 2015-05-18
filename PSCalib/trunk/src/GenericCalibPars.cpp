//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class GenericCalibPars...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSCalib/GenericCalibPars.h"

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

#include "pdscalibdata/CsPadBaseV2.h"
#include "pdscalibdata/CsPad2x2BaseV2.h"
#include "pdscalibdata/PnccdBaseV1.h"
#include "pdscalibdata/PrincetonBaseV1.h" // shape_base(), Ndim, Rows, Cols, Size, etc.
#include "pdscalibdata/AndorBaseV1.h"
#include "pdscalibdata/Epix100aBaseV1.h"
#include "pdscalibdata/VarShapeCameraBaseV1.h"
//#include "pdscalibdata/Opal1000BaseV1.h"
//#include "pdscalibdata/Opal4000BaseV1.h"

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

template <typename TBASE> 
GenericCalibPars<TBASE>::GenericCalibPars ( const std::string&   calibDir,  //  "/reg/d/psdm/AMO/amoa1214/calib"
                                     const std::string&   groupName, //  "PNCCD::CalibV1"
                                     const std::string&   source,    //  "Camp.0:pnCCD.0"
                                     const unsigned long& runNumber, //  10
                                     unsigned             print_bits)
  : PSCalib::CalibPars()
  , TBASE()
  , m_calibDir(calibDir)
  , m_groupName(groupName)
  , m_source(source)
  , m_runNumber(runNumber)
  , m_print_bits(print_bits)
{
    init();
}

//----------------

template <typename TBASE> 
GenericCalibPars<TBASE>::GenericCalibPars ( const std::string&   calibDir,  //  "/reg/d/psdm/AMO/amoa1214/calib"
                                     const std::string&   groupName, //  "PNCCD::CalibV1"
                                     const Pds::Src&      src,       //  Pds::Src m_src; <- is defined in get(...,&m_src)
                                     const unsigned long& runNumber, //  10
                                     unsigned             print_bits)
  : PSCalib::CalibPars()
  , TBASE()
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

template <typename TBASE> 
void GenericCalibPars<TBASE>::init()
{
  m_pedestals    = 0;
  m_pixel_gain   = 0;
  m_pixel_mask   = 0;
  m_pixel_bkgd   = 0;
  m_pixel_rms    = 0;
  m_common_mode  = 0;
  m_pixel_status = 0; 
  m_name = std::string("GenericCalibPars");

  if( m_print_bits & 1 ) printInputPars();
  if( m_print_bits & 16) printCalibTypes(); // method from superclass
  m_prbits_cff  = ( m_print_bits &  2 ) ? 0377 : 0;
  m_prbits_type = ( m_print_bits & 32 ) ? 0377 : 0;
}

//----------------

template <typename TBASE> 
std::string GenericCalibPars<TBASE>::getCalibFileName (const CALIB_TYPE& calibtype)
{
  std::string fname = std::string();
  if (m_calibDir != std::string()) {

    //PSCalib::CalibFileFinder *calibfinder = new PSCalib::CalibFileFinder(m_calibDir, m_groupName, m_prbits_cff);
      PSCalib::CalibFileFinder calibfinder(m_calibDir, m_groupName, m_prbits_cff);

      if (m_source.empty()) // == std::string())
          fname = calibfinder.findCalibFile(m_src, map_type2str[calibtype], m_runNumber);
      else
          fname = calibfinder.findCalibFile(m_source, map_type2str[calibtype], m_runNumber);
  }

  if( m_print_bits & 4 ) MsgLog(m_name, info, "Use calibration parameters from file: " << fname);
  if( m_print_bits & 8 && fname.empty() )
       MsgLog(m_name, warning, "File for calibration type " << map_type2str[calibtype]
	      << " IS MISSING! Use default calibration parameters on your own risk..."); 

  return fname;
}

//----------------

template <typename TBASE> 
const CalibPars::pedestals_t*
GenericCalibPars<TBASE>::pedestals()
{
  if (m_pedestals == 0) {
      std::string fname = getCalibFileName(PEDESTALS);
      //cout << "T: pedestals: " << fname << " size()=" << size() << '\n';
      if (size()) m_pedestals = new NDAIPEDS(fname, shape(), pedestals_t(0), m_prbits_type);
      else        m_pedestals = new NDAIPEDS(fname, m_prbits_type);
  }
  return m_pedestals->get_ndarray().data();
}

//----------------

template <typename TBASE> 
const CalibPars::pixel_status_t*
GenericCalibPars<TBASE>::pixel_status()
{
  if (m_pixel_status == 0) {
      std::string fname = getCalibFileName(PIXEL_STATUS);
      const CalibPars::shape_t* p_sha = shape();
      //cout << "T: pixel_status: " << fname << " size()=" << size() << '\n';
      //cout << "T: shape_nda() : [" << p_sha[0] << "," << p_sha[1] << "]\n";
      if (size()) m_pixel_status = new NDAISTATUS(fname, p_sha, pixel_status_t(1), m_prbits_type);
      else        m_pixel_status = new NDAISTATUS(fname, m_prbits_type);
  }
  return m_pixel_status->get_ndarray().data();
}

//----------------

template <typename TBASE> 
const CalibPars::pixel_gain_t*
GenericCalibPars<TBASE>::pixel_gain()
{
  if (m_pixel_gain == 0) {
      std::string fname = getCalibFileName(PIXEL_GAIN);
      if (size()) m_pixel_gain = new NDAIGAIN(fname, shape(), pixel_gain_t(1), m_prbits_type);
      else        m_pixel_gain = new NDAIGAIN(fname, m_prbits_type);
  }
  return m_pixel_gain->get_ndarray().data();
}

//----------------

template <typename TBASE> 
const CalibPars::pixel_mask_t*
GenericCalibPars<TBASE>::pixel_mask()
{
  if (m_pixel_mask == 0) {
      std::string fname = getCalibFileName(PIXEL_MASK);
      if (size()) m_pixel_mask = new NDAIMASK(fname, shape(), pixel_mask_t(1), m_prbits_type);
      else        m_pixel_mask = new NDAIMASK(fname, m_prbits_type);
  }
  return m_pixel_mask->get_ndarray().data();
}

//----------------

template <typename TBASE> 
const CalibPars::pixel_bkgd_t*
GenericCalibPars<TBASE>::pixel_bkgd()
{
  if (m_pixel_bkgd == 0) {
      std::string fname = getCalibFileName(PIXEL_BKGD);
      if (size()) m_pixel_bkgd = new NDAIBKGD(fname, shape(), pixel_bkgd_t(0), m_prbits_type);
      else        m_pixel_bkgd = new NDAIBKGD(fname, m_prbits_type);
  }
  return m_pixel_bkgd->get_ndarray().data();
}

//----------------

template <typename TBASE> 
const CalibPars::pixel_rms_t*
GenericCalibPars<TBASE>::pixel_rms()
{
  if (m_pixel_rms == 0) {
      std::string fname = getCalibFileName(PIXEL_RMS);
      if (size()) m_pixel_rms = new NDAIRMS(fname, shape(), pixel_rms_t(1), m_prbits_type);
      else        m_pixel_rms = new NDAIRMS(fname, m_prbits_type);
  }
  return m_pixel_rms->get_ndarray().data();
}

//----------------

template <typename TBASE> 
const CalibPars::common_mode_t*
GenericCalibPars<TBASE>::common_mode()
{
  if (m_common_mode == 0) {
      std::string fname = getCalibFileName(COMMON_MODE);
      ndarray<const CalibPars::common_mode_t,1> nda = make_ndarray(TBASE::cmod_base(), TBASE::SizeCM); // see PrincetonBaseV1
      m_common_mode = new NDAICMOD(fname, nda, m_prbits_type);
  }
  return m_common_mode->get_ndarray().data();
}

//----------------

template <typename TBASE> 
const size_t
GenericCalibPars<TBASE>::ndim(const CALIB_TYPE& calibtype)
{ 
  return (calibtype!=COMMON_MODE) ? TBASE::Ndim : 1;
}

//----------------

template <typename TBASE> 
const size_t
GenericCalibPars<TBASE>::size(const CALIB_TYPE& calibtype)
{ 
  const size_t size_nda = (calibtype!=COMMON_MODE) ? size_of_ndarray() : TBASE::SizeCM;
  return (size_nda) ? size_nda : TBASE::size_base();
}

//----------------

template <typename TBASE> 
const CalibPars::shape_t*
//const unsigned*
GenericCalibPars<TBASE>::shape(const CALIB_TYPE& calibtype)
{ 
  const size_t size_nda = size(calibtype);

  if(calibtype==COMMON_MODE) {
    *m_shape_cmode = size_nda;
    return m_shape_cmode;
   }

  return (size_nda) ? shape_of_ndarray() : TBASE::shape_base(); 
}

//----------------

template <typename TBASE> 
const size_t
GenericCalibPars<TBASE>::size_of_ndarray() 
{ 
  if      (m_pedestals   ) return m_pedestals   ->get_ndarray().size();
  else if (m_pixel_status) return m_pixel_status->get_ndarray().size();
  else if (m_pixel_gain  ) return m_pixel_gain  ->get_ndarray().size();
  else if (m_pixel_mask  ) return m_pixel_mask  ->get_ndarray().size();
  else if (m_pixel_bkgd  ) return m_pixel_bkgd  ->get_ndarray().size();
  else if (m_pixel_rms   ) return m_pixel_rms   ->get_ndarray().size();

  if( m_print_bits & 2 ) MsgLog(m_name, warning, "CAN'T RETURN SIZE of non-loaded ndarray"); 
  //return TBASE::Size;
  return TBASE::size_base();
}

//----------------

template <typename TBASE> 
const CalibPars::shape_t*
//const unsigned*
GenericCalibPars<TBASE>::shape_of_ndarray()
{ 
  if      (m_pedestals   ) return m_pedestals   ->get_ndarray().shape();
  else if (m_pixel_status) return m_pixel_status->get_ndarray().shape();
  else if (m_pixel_gain  ) return m_pixel_gain  ->get_ndarray().shape();
  else if (m_pixel_mask  ) return m_pixel_mask  ->get_ndarray().shape();
  else if (m_pixel_bkgd  ) return m_pixel_bkgd  ->get_ndarray().shape();
  else if (m_pixel_rms   ) return m_pixel_rms   ->get_ndarray().shape();
 
  if( m_print_bits & 2 ) MsgLog(m_name, warning, "CAN'T RETURN SHAPE of non-loaded ndarray");
  return TBASE::shape_base();
}

//----------------
//----------------
//----------------
//----------------

template <typename TBASE> 
void GenericCalibPars<TBASE>::printInputPars()
{
    WithMsgLog(m_name, info, str) {
      str << "printInputPars()"
      	  << "\n m_calibDir   = " << m_calibDir 
      	  << "\n m_groupName  = " << m_groupName 
      	  << "\n m_source     = " << m_source 
      	  << "\n m_runNumber  = " << m_runNumber 
      	  << "\n m_print_bits = " << m_print_bits
      	  << "\nDetector base configuration parameters:"
	  << "\n Ndim = " << TBASE::Ndim
          << "\n Shape = [" << str_shape() << "]";
    }
      //std::string str = map_type2str[PEDESTALS];
      //std::cout << "map_type2str[PEDESTALS] = " << str << '\n';
}

//----------------

template <typename TBASE> 
std::string GenericCalibPars<TBASE>::str_shape()
{
    std::stringstream smsg; 
    for (unsigned i=0; i<TBASE::Ndim; i++) {
        if (i) smsg << "," << shape()[i];
        else   smsg << shape()[i];
    }
    return smsg.str();
}

//----------------

template <typename TBASE> 
void GenericCalibPars<TBASE>::loadAllCalibPars ()
{
  pedestals(); 
  pixel_gain();
  pixel_mask();
  pixel_bkgd();
  pixel_rms();
  pixel_status();
  common_mode();
}

//----------------

template <typename TBASE> 
void GenericCalibPars<TBASE>::printCalibParsStatus ()
{
  //loadAllCalibPars ();

    std::stringstream smsg; smsg << "\n  printCalibParsStatus():";
    if (m_pedestals)    smsg << "\n  pedestals    : " << m_pedestals    -> str_status();
    if (m_pixel_status) smsg << "\n  pixel_status : " << m_pixel_status -> str_status();
    if (m_pixel_gain)   smsg << "\n  pixel_gain   : " << m_pixel_gain   -> str_status();
    if (m_pixel_mask)   smsg << "\n  pixel_mask   : " << m_pixel_mask   -> str_status();
    if (m_pixel_bkgd)   smsg << "\n  pixel_bkgd   : " << m_pixel_bkgd   -> str_status();
    if (m_pixel_rms)    smsg << "\n  pixel_rms    : " << m_pixel_rms    -> str_status();
    if (m_common_mode)  smsg << "\n  common_mode  : " << m_common_mode  -> str_status();
    MsgLog(m_name, info, smsg.str());
}

//----------------

template <typename TBASE> 
void GenericCalibPars<TBASE>::printCalibPars()
{
  //loadAllCalibPars ();
    printInputPars();
    printCalibParsStatus ();

    std::stringstream smsg; smsg << "\n  printCalibPars():";
    if (m_pedestals) {  smsg << "\n  shape = ["       << m_pedestals    -> str_shape() << "]"
                             << "\n  pedestals    : " << m_pedestals    -> str_ndarray_info();
    }
    if (m_pixel_status) smsg << "\n  pixel_status : " << m_pixel_status -> str_ndarray_info();
    if (m_pixel_gain)   smsg << "\n  pixel_gain   : " << m_pixel_gain   -> str_ndarray_info();
    if (m_pixel_mask)   smsg << "\n  pixel_mask   : " << m_pixel_mask   -> str_ndarray_info();
    if (m_pixel_bkgd)   smsg << "\n  pixel_bkgd   : " << m_pixel_bkgd   -> str_ndarray_info();
    if (m_pixel_rms)    smsg << "\n  pixel_rms    : " << m_pixel_rms    -> str_ndarray_info();
    if (m_common_mode)  smsg << "\n  common_mode  : " << m_common_mode  -> str_ndarray_info();
    MsgLog(m_name, info, smsg.str());
}

//--------------
// Destructor --
//--------------

template <typename TBASE> 
GenericCalibPars<TBASE>::~GenericCalibPars ()
{
    delete m_pedestals;
    delete m_pixel_status;
    delete m_pixel_gain;
    delete m_pixel_mask;
    delete m_pixel_bkgd;
    delete m_pixel_rms;
    delete m_common_mode;
  //delete [] m_data; 
}

//----------------
//----------------
//----------------
//----------------

template class PSCalib::GenericCalibPars<pdscalibdata::CsPadBaseV2>;
template class PSCalib::GenericCalibPars<pdscalibdata::CsPad2x2BaseV2>;
template class PSCalib::GenericCalibPars<pdscalibdata::PnccdBaseV1>;
template class PSCalib::GenericCalibPars<pdscalibdata::PrincetonBaseV1>;
template class PSCalib::GenericCalibPars<pdscalibdata::AndorBaseV1>;
template class PSCalib::GenericCalibPars<pdscalibdata::Epix100aBaseV1>;
template class PSCalib::GenericCalibPars<pdscalibdata::VarShapeCameraBaseV1>;
//template class PSCalib::GenericCalibPars<pdscalibdata::Opal1000BaseV1>;
//template class PSCalib::GenericCalibPars<pdscalibdata::Opal4000BaseV1>;

//----------------
//----------------
//----------------
//----------------

} // namespace PSCalib
