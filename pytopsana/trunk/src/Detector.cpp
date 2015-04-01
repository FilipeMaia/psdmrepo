//-----------------------
// This Class's Header --
//-----------------------
#include "pytopsana/Detector.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm> // for fill_n

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
// to work with detector data include corresponding 
// header from psddl_psana package
#include "PSEvt/EventId.h"
#include "PSEvt/Source.h"
#include "psddl_psana/pnccd.ddl.h"
#include "PSCalib/CalibPars.h"
//#include "ImgAlgos/GlobalMethods.h" // for DETECTOR_TYPE, getRunNumber(evt), etc.
//#include "PSCalib/CalibParsStore.h"

//-------------------
namespace pytopsana {

  typedef Detector::data_t data_t;
  typedef data_t           TOUT;

  const static size_t   Segs   = 4; 
  const static size_t   Rows   = 512; 
  const static size_t   Cols   = 512; 
  const static size_t   FrSize = Rows*Cols; 
  const static size_t   Size   = Segs*Rows*Cols; 

//----------------
// Constructors --
//----------------
//Detector::Detector (const PSEvt::Source src)
Detector::Detector ()
{
  //m_src = src;
  std::cout << "ctor\n"; // : SOURCE: " << m_src << "\n";
  //m_dettype = ImgAlgos::detectorTypeForSource(m_src);
  //std::cout << "\nCALIB GROUP:" << ImgAlgos::calibGroupForDetType(m_dettype) << '\n';
}

//--------------
// Destructor --
//--------------
Detector::~Detector ()
{
  // Does nothing for now
}

//-------------------

ndarray<data_t,3> Detector::pedestals(PSEvt::Source src, boost::shared_ptr<PSEvt::Event> evt, boost::shared_ptr<PSEnv::Env> env)
{
  m_src = src;
  std::cout << "SOURCE     : " << m_src << '\n';
  std::cout << "INSTRUMENT : " << env->instrument() << '\n';

  initCalibStore(*(evt.get()), *(env.get()));
  
  ndarray<TOUT, 3> ndarr = make_ndarray<TOUT>(Segs, Rows, Cols);    
  std::fill_n(&ndarr[0][0][0], int(Size), TOUT(5));
  return ndarr;
}
  

//-------------------

void 
Detector::initCalibStore(PSEvt::Event& evt, PSEnv::Env& env)
{
  //std::string calib_dir = (m_calibDir == "") ? env.calibDir() : m_calibDir;
  std::string calibdir = env.calibDir();
  boost::shared_ptr<PSEvt::EventId> eventId = evt.get();
  int runnum = (eventId.get()) ? eventId->run() : 0; //ImgAlgos::getRunNumber(evt);
  std::string group = std::string(); // for ex: "PNCCD::CalibV1";
  unsigned prbits = 255;

  std::cout << "\nCalib dir   : " << calibdir 
            << "\nCalib group : " << group 
            << "\nRun number  : " << runnum 
            << "\nPrint bits  : " << prbits 
            << '\n';
  //MsgLog(_name_(), info, "Calibration directory: " << calibdir);

  //PSCalib::CalibPars* m_calibparsm_calibpars = PSCalib::CalibParsStore::Create(calibdir, group, m_src, runnum, prbits);

  //m_peds_data = (m_do_peds) ? m_calibpars->pedestals()    : 0;
  //m_gain_data = (m_do_gain) ? m_calibpars->pixel_gain()   : 0;
  //m_cmod_data = (m_do_cmod) ? m_calibpars->common_mode()  : 0;
  //m_rms_data  = (m_do_nrms) ? m_calibpars->pixel_rms()    : 0;
  //m_stat_data = (m_do_stat || m_do_cmod) ? m_calibpars->pixel_status() : 0;
}

//-------------------
//-------------------
//-------------------
//-------------------
//-------------------
//-------------------
//-------------------
//-------------------


// Return 3D NDarray of raw detector data
ndarray<data_t,3> Detector::raw(PSEvt::Source src, boost::shared_ptr<PSEvt::Event> evt, boost::shared_ptr<PSEnv::Env> env)
{
  std::cout << "SOURCE: " << src << std::endl;
  std::cout << "ENV: " << env->instrument() << std::endl;
  
  boost::shared_ptr<Psana::PNCCD::FramesV1> frames1 = evt->get(src);

  ndarray<TOUT, 3> out_ndarr = make_ndarray<TOUT>(Segs,Rows,Cols);    
  
  if (frames1) {
    
    ndarray<TOUT, 3>::iterator it_out = out_ndarr.begin(); 
            
    for (unsigned i = 0 ; i != frames1->numLinks(); ++ i) {
      
      const Psana::PNCCD::FrameV1& frame = frames1->frame(i);          
      const ndarray<const data_t, 2> data = frame.data();
        
      // Copy frame from data to output ndarray with changing type
      for ( ndarray<const data_t, 2>::iterator it=data.begin(); it!=data.end(); ++it, ++it_out) {
	*it_out = (TOUT)*it;
      }
    }
        
  }

  return out_ndarr;
}


//-------------------

// Return 3D NDarray of calibrated detector data
ndarray<double,3> Detector::calib(PSEvt::Source src, boost::shared_ptr<PSEvt::Event> evt, boost::shared_ptr<PSEnv::Env> env)
{
  ndarray<data_t,3>   raw_data = this->raw(src,evt,env);
  ndarray<double,3> calib_data = this->calib(raw_data);
  return calib_data;
}


//-------------------

ndarray<double,3> Detector::calib(ndarray<data_t,3> raw_data)
{
  // For now, just return a double version of raw_data

  // Create the calib_data array
  ndarray<double,3> calib_data = make_ndarray<double>(Segs,Rows,Cols);

  // Set up iterator to calib_data
  ndarray<double,3>::iterator calib_it = calib_data.begin();

  // Loop over raw_data and copy value as double into calib_data
  for (ndarray<data_t,3>::iterator raw_it = raw_data.begin();
       raw_it != raw_data.end(); raw_it++,calib_it++) {    
    *calib_it = (double) *raw_it;
  }
  
  // return calib_data
  return calib_data;  
}


std::string Detector::env(boost::shared_ptr<PSEnv::Env> env)
{  
  return env->instrument().c_str();
}

//-------------------
} // namespace pytopsana
//-------------------





