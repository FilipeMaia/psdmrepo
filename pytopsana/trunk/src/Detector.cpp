//-----------------------
// This Class's Header --
//-----------------------
#include "pytopsana/Detector.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm> // for fill_n
#include <sstream>  // for stringstream

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
// to work with detector data include corresponding 
// header from psddl_psana package
#include "PSEvt/EventId.h"
#include "PSEvt/Source.h"
#include "psddl_psana/pnccd.ddl.h"

#include "PSCalib/CalibParsStore.h"

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
Detector::Detector (const PSEvt::Source& source, const unsigned& prbits)
  : m_calibpars(0)
  , m_source(source)
  , m_runnum(-1)
  , m_prbits(prbits)
  , m_nda_prod(0)
{
  std::stringstream ss; ss << source;
  m_str_src = ss.str();
  m_dettype = ImgAlgos::detectorTypeForSource(m_source);
  m_cgroup  = ImgAlgos::calibGroupForDetType(m_dettype); // for ex: "PNCCD::CalibV1";

  if(m_prbits) {
      std::stringstream ss;
      ss << "in ctor:" // "SOURCE: " << m_source
  	 << "\nData source  : " << m_str_src
  	 << "\nCalib group  : " << m_cgroup 
         << "\nPrint bits   : " << m_prbits
         << '\n';
      MsgLog(_name_(), info, ss.str());
  }

}

//--------------
// Destructor --
//--------------
Detector::~Detector ()
{
  // Does nothing for now
}

//-------------------

ndarray<const Detector::pedestals_t, 1> Detector::pedestals(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env)
{
  initCalibStore(*shp_evt, *shp_env);
  
  //ndarray<TOUT, 3> ndarr = make_ndarray<TOUT>(Segs, Rows, Cols);    
  //std::fill_n(&ndarr[0][0][0], int(Size), TOUT(5));
  //return ndarr;
  
  //ndarray<const Detector::pedestals_t, 1> nda = make_ndarray(m_calibpars->pedestals(), m_calibpars->size());
  return make_ndarray(m_calibpars->pedestals(), m_calibpars->size());
}

//-------------------

ndarray<const Detector::pixel_rms_t, 1> Detector::pixel_rms(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env)
{
  initCalibStore(*shp_evt, *shp_env);
  return make_ndarray(m_calibpars->pixel_rms(), m_calibpars->size());
}

//-------------------

ndarray<const Detector::pixel_gain_t, 1> Detector::pixel_gain(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env)
{
  initCalibStore(*shp_evt, *shp_env);
  return make_ndarray(m_calibpars->pixel_gain(), m_calibpars->size());
}

//-------------------

ndarray<const Detector::pixel_mask_t, 1> Detector::pixel_mask(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env)
{
  initCalibStore(*shp_evt, *shp_env);
  return make_ndarray(m_calibpars->pixel_mask(), m_calibpars->size());
}

//-------------------

ndarray<const Detector::pixel_bkgd_t, 1> Detector::pixel_bkgd(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env)
{
  initCalibStore(*shp_evt, *shp_env);
  return make_ndarray(m_calibpars->pixel_bkgd(), m_calibpars->size());
}

//-------------------

ndarray<const Detector::pixel_status_t, 1> Detector::pixel_status(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env)
{
  initCalibStore(*shp_evt, *shp_env);
  return make_ndarray(m_calibpars->pixel_status(), m_calibpars->size());
}

//-------------------

ndarray<const Detector::common_mode_t, 1> Detector::common_mode(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env)
{
  initCalibStore(*shp_evt, *shp_env);
  //std::cout << "TEST cm[0]: " << m_calibpars->common_mode()[0] << "\n";
  //std::cout << "TEST cm[3]: " << m_calibpars->common_mode()[3] << "\n";
  //std::cout << "TEST  size: " << m_calibpars->size(PSCalib::COMMON_MODE) << "\n";
  return make_ndarray(m_calibpars->common_mode(), m_calibpars->size(PSCalib::COMMON_MODE));
}

//-------------------

void 
Detector::initCalibStore(PSEvt::Event& evt, PSEnv::Env& env)
{
  int runnum = ImgAlgos::getRunNumber(evt);
  if(runnum == m_runnum) return;
  m_runnum = runnum;

  if(m_calibpars) delete m_calibpars;

  //std::string calib_dir = (m_calibDir == "") ? env.calibDir() : m_calibDir;
  std::string calibdir = env.calibDir();

  if(m_prbits) {
      std::stringstream ss;
      ss << "in initCalibStore(...):"
         << "\nInstrument  : " << env.instrument()
         << "\nCalib dir   : " << calibdir 
         << "\nCalib group : " << m_cgroup 
         << "\nData source : " << m_str_src
         << "\nRun number  : " << m_runnum 
         << "\nPrint bits  : " << m_prbits 
         << '\n';
      
      MsgLog(_name_(), info, ss.str());
  }

  m_calibpars = PSCalib::CalibParsStore::Create(calibdir, m_cgroup, m_str_src, m_runnum, m_prbits);

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

void 
Detector::initNDArrProducer(const PSEvt::Source& source)
{
  //if(!m_nda_prod) m_nda_prod = new NDArrProducerCSPAD(source);  // direct access
  if(!m_nda_prod) m_nda_prod = NDArrProducerStore::Create(source); // universal access through the factory store
}

//-------------------
//-------------------

ndarray<const int16_t, 1> Detector::data_int16_1(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env)
{
  initNDArrProducer(m_source);
  return m_nda_prod->data_nda_int16_1(*shp_evt, *shp_env);
}

//-------------------

ndarray<const int16_t, 2> Detector::data_int16_2(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env)
{
  initNDArrProducer(m_source);
  return m_nda_prod->data_nda_int16_2(*shp_evt, *shp_env);
}

//-------------------

ndarray<const int16_t, 3> Detector::data_int16_3(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env)
{
  initNDArrProducer(m_source);
  //return m_nda_prod->getNDArr(*shp_evt, *shp_env);
  return m_nda_prod->data_nda_int16_3(*shp_evt, *shp_env);
}

//-------------------

ndarray<const int16_t, 4> Detector::data_int16_4(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env)
{
  initNDArrProducer(m_source);
  return m_nda_prod->data_nda_int16_4(*shp_evt, *shp_env);
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
ndarray<data_t,3> Detector::raw(PSEvt::Source source, boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env)
{
  std::cout << "SOURCE: " << source << std::endl;
  std::cout << "ENV: " << shp_env->instrument() << std::endl;
  
  boost::shared_ptr<Psana::PNCCD::FramesV1> frames1 = shp_evt->get(source);

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
ndarray<double,3> Detector::calib(PSEvt::Source source, boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env)
{
  ndarray<data_t,3>   raw_data = this->raw(source,shp_evt,shp_env);
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


std::string Detector::str_inst(boost::shared_ptr<PSEnv::Env> shp_env)
{  
  return shp_env->instrument().c_str();
}

//-------------------
} // namespace pytopsana
//-------------------





