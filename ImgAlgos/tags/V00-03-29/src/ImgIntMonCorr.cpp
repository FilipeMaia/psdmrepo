//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgIntMonCorr...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/ImgIntMonCorr.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream> // for stringstream
#include <iomanip> // for setw, setprecision, left, right manipulators

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"
#include "psddl_psana/bld.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(ImgIntMonCorr)

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
ImgIntMonCorr::ImgIntMonCorr (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key_in()
  , m_key_out() 
  , m_fname_imon_cfg()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_str_src           = configSrc("source",   "DetInfo(:Camera)");
  m_key_in            = configStr("key_in",         "calibrated");
  m_key_out           = configStr("key_out",    "imon_corrected");
  m_fname_imon_cfg    = configStr("fname_imon_cfg",           "");
  m_print_bits        = config   ("print_bits",               0 );

  m_do_sele = false;
  m_do_norm = false;
}

//--------------------

void 
ImgIntMonCorr::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters  :"
        << "\n source            : " << m_str_src
        << "\n m_key_in          : " << m_key_in      
        << "\n m_key_out         : " << m_key_out
        << "\n m_fname_imon_cfg  : " << m_fname_imon_cfg
        << "\n m_do_sele         : " << m_do_sele
        << "\n m_do_norm         : " << m_do_norm     
        << "\n m_print_bits      : " << m_print_bits
        << "\n";     
  }
}

//--------------------


//--------------
// Destructor --
//--------------
ImgIntMonCorr::~ImgIntMonCorr ()
{
}

/// Method which is called once at the beginning of the job
void 
ImgIntMonCorr::beginJob(Event& evt, Env& env)
{
  readIntMonConfigFile();  
  if( m_print_bits & 1 ) printIntMonConfig(); 
  if( m_print_bits & 1 ) printInputParameters();
}

/// Method which is called at the beginning of the run
void 
ImgIntMonCorr::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
ImgIntMonCorr::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ImgIntMonCorr::event(Event& evt, Env& env)
{
  if(!m_count) init(evt, env);
  if( m_print_bits & 2 ) printEventRecord(evt);
  if( m_print_bits & 4 ) printIntMonData(evt, env);

  m_norm_factor = 1;

  if (m_do_sele or m_do_norm) {
     if( !procIntMonData(evt, env) ) { skip(); return; } // event is discarded
  }

  // m_norm_factor = 1; // for test only

  if( m_print_bits & 32 ) printNormFactor();

  procEvent(evt, env);
  ++ m_count;
}
  
/// Method which is called at the end of the calibration cycle
void 
ImgIntMonCorr::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
ImgIntMonCorr::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
ImgIntMonCorr::endJob(Event& evt, Env& env)
{
}

//--------------------

void 
ImgIntMonCorr::init(Event& evt, Env& env)
{
    defineImageShape(evt, m_str_src, m_key_in, m_shape); // shape is not available in beginJob and beginRun
    m_rows = m_shape[0];
    m_cols = m_shape[1];
    m_size = m_rows*m_cols;

    //m_cdat = new double [m_size];
    //std::fill_n(m_cdat, int(m_size), 0);    
}

//--------------------

void 
ImgIntMonCorr::procEvent(Event& evt, Env& env)
{
  if ( procEventForType<double>   (evt) ) return;
  if ( procEventForType<uint16_t> (evt) ) return;
  if ( procEventForType<int>      (evt) ) return;
  if ( procEventForType<float>    (evt) ) return;
  if ( procEventForType<uint8_t>  (evt) ) return;

  MsgLog(name(), info, "Image is not available in the event(...) for source:" << m_str_src << " key:" << m_key_in);
}

//--------------------

void 
ImgIntMonCorr::printEventRecord(Event& evt)
{
  MsgLog( name(), info,  "Run="    << stringRunNumber(evt) 
                     << " Evt="    << stringFromUint(m_count) 
                     << " Time="   << stringTimeStamp(evt) 
  );
}

//--------------------

void 
ImgIntMonCorr::printNormFactor()
{
  MsgLog( name(), info, "Intensity norm. factor: " << m_norm_factor);
}

//--------------------

void
ImgIntMonCorr::readIntMonConfigFile()
{
  MsgLog( name(), info,  "ImgIntMonCorr::readIntMonConfigFile(): Read intensity monitor configuration from file: " << m_fname_imon_cfg << "\n");
  
  std::string s;
  IntMonConfig imc;

  std::ifstream inf(m_fname_imon_cfg.c_str());
  if (inf.is_open())
  {
    while ( true )
    {
      getline (inf,s);
      if(!inf.good()) break;
      std::stringstream ss(s); 
      ss >> imc.src_name  >> imc.name >> imc.ch1 >> imc.ch2 >> imc.ch3 >> imc.ch4 >> imc.norm >> imc.sele >> imc.imin >> imc.imax >> imc.iave;
      imc.source = Source(imc.src_name);
      v_imon_cfg.push_back(imc);

      if (imc.sele) m_do_sele = true;
      if (imc.norm) m_do_norm = true;
    }
    inf.close();
  }
  else {
    MsgLog( name(), info,  "ImgIntMonCorr::readIntMonConfigFile(): Unable to open file: " << m_fname_imon_cfg << "\n"); 
    abort(); 
  }
}

//--------------------

void
ImgIntMonCorr::printIntMonConfig()
{
  WithMsgLog(name(), info, log) {
      log << "ImgIntMonCorr::printIntMonConfig(): Intensity monitor configuration from file: " 
          << m_fname_imon_cfg << " size=" << v_imon_cfg.size() << "\n";

    for(std::vector<IntMonConfig>::const_iterator it = v_imon_cfg.begin(); 
                                                  it!= v_imon_cfg.end(); it++) {

                     // << std::fixed << std::setw(15) << std::setprecision(3)
      log << " source:" << std::setw(32) << std::left  << it->src_name // source
          << " name:"   << std::setw(16)               << it->name 
          << " ch1:"    << std::setw(1)  << std::right << it->ch1   
          << " ch2:"    << std::setw(1)                << it->ch2   
          << " ch3:"    << std::setw(1)                << it->ch3   
          << " ch4:"    << std::setw(1)                << it->ch4   
          <<"  norm:"   << std::setw(1)                << it->norm  
          << " sele:"   << std::setw(1)                << it->sele  
          <<"  imin:"   << std::setw(8)                << it->imin  
          << " imax:"   << std::setw(8)                << it->imax  
          << " iave:"   << std::setw(8)                << it->iave  
          << "\n";      
    }
    log << "\n";
  }
}

//--------------------

void
ImgIntMonCorr::printIntMonData(Event& evt, Env& env)
{
  WithMsgLog(name(), info, log) {
    for(std::vector<IntMonConfig>::const_iterator it = v_imon_cfg.begin(); 
                                                  it!= v_imon_cfg.end(); it++) {

      Quartet q = getIntMonDataForSource(evt, env, it->source);
      log << "\nIntensity monitor data for source " << std::setw(32) << std::left << it->src_name
          << ": " << std::setw(8) << std::setprecision(4) << q.v1
          << " "  << std::setw(8) << std::setprecision(4) << q.v2
          << " "  << std::setw(8) << std::setprecision(4) << q.v3
          << " "  << std::setw(8) << std::setprecision(4) << q.v4;
    }
    //log << "\n";
  }
}

//--------------------
// 1) Get intensity monitor data
// 2) Apply selector
// 3) Evaluate correction factor
bool 
ImgIntMonCorr::procIntMonData(Event& evt, Env& env)
{
    for(std::vector<IntMonConfig>::const_iterator it = v_imon_cfg.begin(); 
                                                  it!= v_imon_cfg.end(); it++) {

      Quartet q = getIntMonDataForSource(evt, env, it->source);
      double sum_int = q.v1 * it->ch1
                     + q.v2 * it->ch2
                     + q.v3 * it->ch3
	             + q.v4 * it->ch4;

      if (it->sele) {
	if(sum_int < it->imin) return false;
	if(sum_int > it->imax) return false;
      }

      if (it->norm) {
        m_norm_factor = (sum_int>0) ? it->iave/sum_int : 1.;
      }
    }
  return true;
}

//--------------------

Quartet
ImgIntMonCorr::getIntMonDataForSource(Event& evt, Env& env, const Source& src)
{  
  shared_ptr<Psana::Bld::BldDataFEEGasDetEnergy> fee = evt.get(src);
  if (fee.get()) {

    if( m_print_bits & 64 ) MsgLog( name(), info, "get: " << src << " from Bld::BldDataFEEGasDetEnergy, f_ij_ENRC(): " 
				    << " " << fee->f_11_ENRC()
				    << " " << fee->f_12_ENRC()
				    << " " << fee->f_21_ENRC()
				    << " " << fee->f_22_ENRC()
    );

    return Quartet ((float)fee->f_11_ENRC(), 
                    (float)fee->f_12_ENRC(), 
                    (float)fee->f_21_ENRC(), 
                    (float)fee->f_22_ENRC());
  } 

//----- for XCS-IPM-02, XCS-IPM-mono, XcsBeamline.1:Ipimb.4, XcsBeamline.1:Ipimb.5, see: psana_examples/src/DumpIpimb.cpp

  shared_ptr<Psana::Lusi::IpmFexV1> fex = evt.get(src);
  if (fex) {

    if( m_print_bits & 64 ) MsgLog( name(), info, "get: " << src << " from Lusi::IpmFexV1" << " channel =" << fex->channel() );

    return Quartet ((float)fex->channel()[0], 
                    (float)fex->channel()[1], 
                    (float)fex->channel()[2], 
                    (float)fex->channel()[3]);
  }

 
  shared_ptr<Psana::Ipimb::DataV2> data2 = evt.get(src);
  if (data2.get()) {

    if( m_print_bits & 64 ) MsgLog( name(), info, "get: " << src << " from Ipimb::DataV2" );

    return Quartet ((float)data2->channel0Volts(), 
                    (float)data2->channel1Volts(), 
                    (float)data2->channel2Volts(), 
                    (float)data2->channel3Volts());
  }

//----- for XCS-IPM-02 and XCS-IPM-mono, see psana_examples/src/DumpBld.cpp

  shared_ptr<Psana::Bld::BldDataIpimbV1> ipimb1 = evt.get(src); 
  if (ipimb1.get()) {

    const Psana::Lusi::IpmFexV1& ipmFexData = ipimb1->ipmFexData();

    if( m_print_bits & 64 ) MsgLog( name(), info, "get: " << src << " from BldDataIpimbV1" << ipmFexData.channel() );

    return Quartet ((float)ipmFexData.channel()[0], 
                    (float)ipmFexData.channel()[1], 
                    (float)ipmFexData.channel()[2], 
                    (float)ipmFexData.channel()[3]);
    /*
    const Psana::Ipimb::DataV2& ipimbData = ipimb1->ipimbData();
    return Quartet ((float)ipimbData.channel0Volts(), 
                    (float)ipimbData.channel1Volts(), 
                    (float)ipimbData.channel2Volts(), 
                    (float)ipimbData.channel3Volts());
    */
  }

//-----
  MsgLog( name(), info,  "IntensityMonitorsData::getDataForSource(): unavailable data for source: " << src << "\n"); 
  //abort(); 
  return  Quartet (-1,-1,-1,-1);
}

//--------------------

Quartet
ImgIntMonCorr::getIntMonDataForSourceV1(Event& evt, Env& env, const Source& src)
{  
  shared_ptr<Psana::Bld::BldDataFEEGasDetEnergy> fee = evt.get(src);
  if (fee.get()) {
    return Quartet ((float)fee->f_11_ENRC(), 
                    (float)fee->f_12_ENRC(), 
                    (float)fee->f_21_ENRC(), 
                    (float)fee->f_22_ENRC());
  } 

//-----
 
  shared_ptr<Psana::Ipimb::DataV2> data2 = evt.get(src);
  if (data2.get()) {
    return Quartet ((float)data2->channel0Volts(), 
                    (float)data2->channel1Volts(), 
                    (float)data2->channel2Volts(), 
                    (float)data2->channel3Volts());
  }

//-----

  shared_ptr<Psana::Bld::BldDataIpimbV1> ipimb1 = evt.get(src);
  if (ipimb1.get()) {
    const Psana::Ipimb::DataV2& ipimbData = ipimb1->ipimbData();

    return Quartet ((float)ipimbData.channel0Volts(), 
                    (float)ipimbData.channel1Volts(), 
                    (float)ipimbData.channel2Volts(), 
                    (float)ipimbData.channel3Volts());
  }

//-----
  MsgLog( name(), info,  "ImgIntMonCorr::getIntMonDataForSource(): unavailable data for source: " << src << "\n"); 
  //abort(); 
  return  Quartet (-1,-1,-1,-1);
}

//--------------------
//--------------------
} // namespace ImgAlgos
