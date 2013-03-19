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

  m_do_sele = (m_fname_imon_cfg.empty()) ? false : true;
  m_do_norm = (m_fname_imon_cfg.empty()) ? false : true;
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

  m_norm_factor = 1;

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
  if ( procEventForType<uint16_t> (evt) ) return;
  if ( procEventForType<int>      (evt) ) return;
  if ( procEventForType<float>    (evt) ) return;
  if ( procEventForType<uint8_t>  (evt) ) return;
  if ( procEventForType<double>   (evt) ) return;

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
      ss >> imc.ind >> imc.source  >> imc.name >> imc.ch1 >> imc.ch2 >> imc.ch3 >> imc.ch4 >> imc.norm >> imc.sele >> imc.imin >> imc.imax >> imc.iave;
      v_imon_cfg.push_back(imc);
    }
    inf.close();
  }
  else MsgLog( name(), info,  "ImgIntMonCorr::readIntMonConfigFile(): Unable to open file: " << m_fname_imon_cfg << "\n");  
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

        //<< " t_sec:"  << std::fixed << std::setw(15) << std::setprecision(3) << it->t_sec
      log << " ind:"    << std::setw(2)  << std::right << it->ind
          << " source:" << std::setw(32) << std::left  << it->source
          << " name:"   << std::setw(16)               << it->name 
          << " ch1:"    << std::setw(1)  << std::right << it->ch1   
          << " ch2:"    << std::setw(1)                << it->ch2   
          << " ch3:"    << std::setw(1)                << it->ch3   
          << " ch4:"    << std::setw(1)                << it->ch4   
          << " norm:"   << std::setw(1)                << it->norm  
          << " sele:"   << std::setw(1)                << it->sele  
          << " imin:"   << std::setw(8)                << it->imin  
          << " imax:"   << std::setw(8)                << it->imax  
          << " iave:"   << std::setw(8)                << it->iave  
          << "\n";      
    }
    log << "\n";
  }
}

//--------------------

//--------------------
//--------------------
} // namespace ImgAlgos
