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
#include "PSEvt/EventId.h"
#include "PSEvt/Source.h"
#include "PSCalib/CalibParsStore.h"

//-------------------
namespace pytopsana {

  typedef Detector::data_t data_t;

//----------------
// Constructors --
//----------------
Detector::Detector (const PSEvt::Source& source, const unsigned& pbits)
  : m_calibpars(0)
  , m_source(source)
  , m_runnum(-1)
  , m_mode(0)
  , m_pbits(pbits)
  , m_vdef(0)
  , m_nda_prod(0)
{
  std::stringstream ss; ss << source;
  m_str_src = ss.str();
  m_dettype = ImgAlgos::detectorTypeForSource(m_source);
  m_cgroup  = ImgAlgos::calibGroupForDetType(m_dettype); // for ex: "PNCCD::CalibV1";
  m_calibdir = std::string();

  if(m_pbits) {
      std::stringstream ss;
      ss << "in ctor:" // "SOURCE: " << m_source
  	 << "\nData source  : " << m_str_src
  	 << "\nCalib group  : " << m_cgroup 
         << "\nPrint bits   : " << m_pbits
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
//-------------------
//-------------------
//-------------------

void 
Detector::initCalibStore(PSEvt::Event& evt, PSEnv::Env& env)
{
  int runnum = ImgAlgos::getRunNumber(evt);
  if(runnum == m_runnum) return;
  m_runnum = runnum;

  if(m_calibpars) delete m_calibpars;

  m_calibdir = env.calibDir();

  if(m_pbits) {
      std::stringstream ss;
      ss << "in initCalibStore(...):"
         << "\nInstrument  : " << env.instrument()
         << "\nCalib dir   : " << m_calibdir 
         << "\nCalib group : " << m_cgroup 
         << "\nData source : " << m_str_src
         << "\nRun number  : " << m_runnum 
         << "\nPrint bits  : " << m_pbits 
         << '\n';
      
      MsgLog(_name_(), info, ss.str());
  }

  m_calibpars = PSCalib::CalibParsStore::Create(m_calibdir, m_cgroup, m_str_src, m_runnum, m_pbits);
}

//-------------------
//-------------------

void 
Detector::initNDArrProducer(const PSEvt::Source& source)
{
  if(!m_nda_prod) m_nda_prod = NDArrProducerStore::Create(source, m_mode, m_pbits, m_vdef); // universal access through the factory store
}

//-------------------
//-------------------
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
  return m_nda_prod->data_nda_int16_3(*shp_evt, *shp_env);
}

//-------------------

ndarray<const int16_t, 4> Detector::data_int16_4(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env)
{
  initNDArrProducer(m_source);
  return m_nda_prod->data_nda_int16_4(*shp_evt, *shp_env);
}

//-------------------

ndarray<const uint16_t, 2> Detector::data_uint16_2(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env)
{
  initNDArrProducer(m_source);
  return m_nda_prod->data_nda_uint16_2(*shp_evt, *shp_env);
}

//-------------------

ndarray<const uint16_t, 3> Detector::data_uint16_3(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env)
{
  initNDArrProducer(m_source);
  return m_nda_prod->data_nda_uint16_3(*shp_evt, *shp_env);
}

//-------------------

ndarray<const uint8_t, 2> Detector::data_uint8_2(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env)
{
  initNDArrProducer(m_source);
  return m_nda_prod->data_nda_uint8_2(*shp_evt, *shp_env);
}

//-------------------

void Detector::print()
{
  initNDArrProducer(m_source);
  return m_nda_prod->print();
}

//-------------------

void Detector::print_config(boost::shared_ptr<PSEvt::Event> shp_evt, boost::shared_ptr<PSEnv::Env> shp_env)
{
  initNDArrProducer(m_source);
  return m_nda_prod->print_config(*shp_evt, *shp_env);
}

//-------------------

std::string Detector::str_inst(boost::shared_ptr<PSEnv::Env> shp_env)
{  
  return shp_env->instrument().c_str();
}

//-------------------
} // namespace pytopsana
//-------------------





