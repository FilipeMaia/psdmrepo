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

  // Default constructor
  Detector (); //const PSEvt::Source src) ;
  
  // Destructor
  virtual ~Detector () ;

  ndarray<data_t,3> pedestals(PSEvt::Source src, boost::shared_ptr<PSEvt::Event> evt, boost::shared_ptr<PSEnv::Env> env);
  
//-------------------
//-------------------
//-------------------
//-------------------

  // Return 3D NDarray of raw detector data
  ndarray<data_t,3> raw(PSEvt::Source src, boost::shared_ptr<PSEvt::Event> evt, boost::shared_ptr<PSEnv::Env> env);  
  
  // Return 3D NDarray of calibrated detector data
  ndarray<double,3> calib(PSEvt::Source src, boost::shared_ptr<PSEvt::Event> evt, boost::shared_ptr<PSEnv::Env> env);
  ndarray<double,3> calib(ndarray<data_t,3> raw_data);

  // Returns instrument as string
  std::string env(boost::shared_ptr<PSEnv::Env> env);  
  
 private:

   PSEvt::Source       m_src;
   //ImgAlgos::DETECTOR_TYPE       m_dettype;          // numerated detector type source
   //PSCalib::CalibPars* m_calibpars;        // pointer to calibration store
 
   //   const PSCalib::CalibPars::pedestals_t*     m_peds_data;
   //   const PSCalib::CalibPars::pixel_gain_t*    m_gain_data;
   //   const PSCalib::CalibPars::pixel_status_t*  m_stat_data;
   //   const PSCalib::CalibPars::common_mode_t*   m_cmod_data;
   //   const PSCalib::CalibPars::pixel_rms_t*     m_rms_data;

   void initCalibStore(PSEvt::Event& evt, PSEnv::Env& env);
   inline const char* _name_(){return "Detector";}
};

//-------------------
} // namespace pytopsana
//-------------------

#endif // PYTOPSANA_DETECTOR_H
