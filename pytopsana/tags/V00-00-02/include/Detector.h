#ifndef PYTOPSANA_DETECTOR_H
#define PYTOPSANA_DETECTOR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Detector
//
//------------------------------------------------------------------------

//-----------------
// Headers --
//-----------------

#include <string>
#include <PSEvt/Event.h>
#include <PSEnv/Env.h>

#include "ndarray/ndarray.h"
#include <boost/shared_ptr.hpp>

#include "ImgAlgos/GlobalMethods.h" // for DETECTOR_TYPE, getRunNumber(evt), etc.
#include "PSCalib/CalibPars.h"
#include "pytopsana/NDArrProducerStore.h"
//#include "pytopsana/NDArrProducerCSPAD.h"

//-------------------
namespace pytopsana {

/**
 *  @defgroup pytopsana package 
 *  @brief Package pytopsana contains classes to call c++ classes/methods from python.
 *
 */

/// @addtogroup pytopsana

/**
 *  @ingroup pytopsana
 *
 *  @brief Class helps to get data/calibrations associated with detector.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class Detector {
 public:

  typedef uint16_t data_t;

  typedef PSCalib::CalibPars::pedestals_t     pedestals_t;
  typedef PSCalib::CalibPars::pixel_rms_t     pixel_rms_t;
  typedef PSCalib::CalibPars::pixel_gain_t    pixel_gain_t;
  typedef PSCalib::CalibPars::pixel_mask_t    pixel_mask_t;
  typedef PSCalib::CalibPars::pixel_bkgd_t    pixel_bkgd_t;
  typedef PSCalib::CalibPars::pixel_status_t  pixel_status_t;
  typedef PSCalib::CalibPars::common_mode_t   common_mode_t;

  typedef NDArrProducerCSPAD::data_t          data_i16_t;  

  // Default constructor
  Detector (const PSEvt::Source& source, const unsigned& prbits=0x0) ; // 0xffff
  
  // Destructor
  virtual ~Detector () ;

  ndarray<const pedestals_t,1>    pedestals   (boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env);
  ndarray<const pixel_rms_t,1>    pixel_rms   (boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env);
  ndarray<const pixel_gain_t,1>   pixel_gain  (boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env);
  ndarray<const pixel_mask_t,1>   pixel_mask  (boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env);
  ndarray<const pixel_bkgd_t,1>   pixel_bkgd  (boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env);
  ndarray<const pixel_status_t,1> pixel_status(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env);
  ndarray<const common_mode_t,1>  common_mode (boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env);
  
//-------------------

  ndarray<const int16_t, 1>       data_int16_1(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env);
  ndarray<const int16_t, 2>       data_int16_2(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env);
  ndarray<const int16_t, 3>       data_int16_3(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env);
  ndarray<const int16_t, 4>       data_int16_4(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env);

//-------------------
//-------------------
//-------------------

  // Return 3D NDarray of raw detector data
  ndarray<data_t,3> raw(PSEvt::Source source, boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env);  
  
  // Return 3D NDarray of calibrated detector data
  ndarray<double,3> calib(PSEvt::Source source, boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env);
  ndarray<double,3> calib(ndarray<data_t,3> raw_data);

  // Returns instrument as string
  std::string str_inst(boost::shared_ptr<PSEnv::Env> shp_env);  
  
 private:

   ImgAlgos::DETECTOR_TYPE m_dettype;          // numerated detector type source
   PSCalib::CalibPars*     m_calibpars;        // pointer to calibration store
   PSEvt::Source           m_source;
   std::string             m_str_src;
   std::string             m_cgroup;
   int                     m_runnum;
   unsigned                m_prbits; 

   //NDArrProducerCSPAD*     m_nda_prod;
   NDArrProducerBase*      m_nda_prod;

   //   const PSCalib::CalibPars::pedestals_t*     m_peds_data;
   //   const PSCalib::CalibPars::pixel_gain_t*    m_gain_data;
   //   const PSCalib::CalibPars::pixel_mask_t*    m_mask_data;
   //   const PSCalib::CalibPars::pixel_bkgd_t*    m_bkgd_data;
   //   const PSCalib::CalibPars::pixel_status_t*  m_stat_data;
   //   const PSCalib::CalibPars::common_mode_t*   m_cmod_data;
   //   const PSCalib::CalibPars::pixel_rms_t*     m_rms_data;

   //const pedestals_t* m_peds_data;

   inline const char* _name_(){return "Detector";}

   void initCalibStore(PSEvt::Event& evt, PSEnv::Env& env);
   void initNDArrProducer(const PSEvt::Source& source);
};

//-------------------
} // namespace pytopsana
//-------------------

#endif // PYTOPSANA_DETECTOR_H
