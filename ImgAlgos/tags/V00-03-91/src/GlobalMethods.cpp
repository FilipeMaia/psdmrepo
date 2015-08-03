//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class GlobalMethods...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/GlobalMethods.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/lexical_cast.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include <boost/shared_ptr.hpp>
#include "PSEvt/EventId.h"
#include "PSTime/Time.h"
//#include "psana/Module.h"
#include "pdsdata/xtc/DetInfo.hh" // for srcToString( const Pds::Src& src )
#include "PSCalib/Exceptions.h"   // for srcToString( const Pds::Src& src )

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace std;
using namespace ImgAlgos;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//--------------------
//--------------------

NDArrPars::NDArrPars()
  : m_ndim(0)
  , m_size(0)
  , m_dtype(DOUBLE)
  , m_src()
  , m_is_set(false)
{
  //m_shape[0] = 0;  
  for(unsigned i=0; i<5; ++i) m_shape[i] = 0;
}

NDArrPars::NDArrPars(const unsigned ndim, const unsigned size, const unsigned* shape, const DATA_TYPE dtype, const Pds::Src& src)
{
  setPars(ndim, size, shape, dtype, src);
}

void
NDArrPars::setPars(const unsigned ndim, const unsigned size, const unsigned* shape, const DATA_TYPE dtype, const Pds::Src& src)
{
  m_ndim   = ndim;
  m_size   = size;
  m_dtype  = dtype;
  m_src    = src;
  m_is_set = true;
  for(unsigned i=0; i<m_ndim; ++i) m_shape[i] = shape[i];
}

void
NDArrPars::print()
{
  WithMsgLog("NDArrPars::print():", info, log) {
    log << "\n NDArrPars parameters:"
        << "\n ndim   : " << m_ndim 
        << "\n size   : " << m_size    
        << "\n dtype  : " << m_dtype  
        << "\n src    : " << ImgAlgos::srcToString(m_src)
        << "\n is_set : " << m_is_set 
        << "\n shape  : ";                        
    for(unsigned i=0; i<m_ndim; ++i) log << m_shape[i] << " ";
    log << "\n";
  }
}

//--------------------
//--------------------

std::string 
stringFromUint(unsigned number, unsigned width, char fillchar)
{
  stringstream ssNum; ssNum << setw(width) << setfill(fillchar) << number;
  return ssNum.str();
}

//--------------------

std::string
stringTimeStamp(PSEvt::Event& evt, std::string fmt) // fmt="%Y%m%dT%H:%M:%S%f"
{
  boost::shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) return (eventId->time()).asStringFormat(fmt);
  else               return std::string("time-stamp-is-unavailable");
}

//--------------------

double
doubleTime(PSEvt::Event& evt)
{
  boost::shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) return double(eventId->time().sec()) + 1e-9*eventId->time().nsec();
  else               return double(0);
}

//--------------------

std::string  
stringRunNumber(PSEvt::Event& evt, unsigned width)
{
  boost::shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) return stringFromUint(eventId->run(), width);
  else               return std::string("run-is-not-defined");
}

//--------------------

/// Returns integer run number
int 
getRunNumber(PSEvt::Event& evt)
{
  boost::shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    return eventId->run();
  } else {
    MsgLogRoot(warning, "Cannot determine run number, will use 0.");
    return int(0);
  }
}

//--------------------

unsigned 
fiducials(PSEvt::Event& evt)
{
  boost::shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) return eventId->fiducials();
  else               return 0;
}

//--------------------

unsigned 
eventCounterSinceConfigure(PSEvt::Event& evt)
{
  boost::shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) return eventId->vector();
  else               return 0;
}

//--------------------

void 
printSizeOfTypes()
{
  std::cout << "Size Of Types:" 
            << "\nsizeof(bool    ) = " << sizeof(bool    ) << " with typeid(bool    ).name(): " << typeid(bool    ).name() 
            << "\nsizeof(int     ) = " << sizeof(int     ) << " with typeid(int     ).name(): " << typeid(int     ).name()  
            << "\nsizeof(float   ) = " << sizeof(float   ) << " with typeid(float   ).name(): " << typeid(float   ).name()  
            << "\nsizeof(double  ) = " << sizeof(double  ) << " with typeid(double  ).name(): " << typeid(double  ).name()  
            << "\nsizeof(short   ) = " << sizeof(short   ) << " with typeid(short   ).name(): " << typeid(short   ).name()  
            << "\nsizeof(uint8_t ) = " << sizeof(uint8_t ) << " with typeid(uint8_t ).name(): " << typeid(uint8_t ).name()  
            << "\nsizeof(uint16_t) = " << sizeof(uint16_t) << " with typeid(uint16_t).name(): " << typeid(uint16_t).name()  
            << "\nsizeof(uint32_t) = " << sizeof(uint32_t) << " with typeid(uint32_t).name(): " << typeid(uint32_t).name()  
            << "\nsizeof(int16_t ) = " << sizeof(int16_t ) << " with typeid(int16_t ).name(): " << typeid(int16_t ).name()  
            << "\nsizeof(int32_t ) = " << sizeof(int32_t ) << " with typeid(int32_t ).name(): " << typeid(int32_t ).name()  
            << "\nsizeof(unsigned short) = " << sizeof(unsigned short) << " with typeid(unsigned short).name(): " << typeid(unsigned short).name()  
            << "\n\n";
}

//--------------------
// Define the shape or throw message that can not do that.
bool
defineImageShape(PSEvt::Event& evt, const PSEvt::Source& src, const std::string& key, unsigned* shape)
{
  if ( defineImageShapeForType<double>  (evt, src, key, shape) ) return true;
  if ( defineImageShapeForType<float>   (evt, src, key, shape) ) return true;
  if ( defineImageShapeForType<int>     (evt, src, key, shape) ) return true;
  if ( defineImageShapeForType<int32_t> (evt, src, key, shape) ) return true;
  if ( defineImageShapeForType<uint32_t>(evt, src, key, shape) ) return true;
  if ( defineImageShapeForType<uint16_t>(evt, src, key, shape) ) return true;
  if ( defineImageShapeForType<uint8_t> (evt, src, key, shape) ) return true;
  if ( defineImageShapeForType<int16_t> (evt, src, key, shape) ) return true;
  if ( defineImageShapeForType<short>   (evt, src, key, shape) ) return true;


  static long counter = 0; counter ++;

  if (counter < 11) {
    const std::string msg = "Image shape is tested for double, uint16_t, int, float, uint8_t, int16_t, short and is not defined in the event(...)\nfor source:" 
                          + boost::lexical_cast<std::string>(src) + " key:" + key;
    MsgLog("GlobalMethods::defineImageShape", warning, msg);
    if (counter == 10) MsgLog("GlobalMethods::defineImageShape", warning, "STOP PRINT WARNINGS for source:" 
              << boost::lexical_cast<std::string>(src) << " key:" << key);
  //throw std::runtime_error("EXIT psana...");
  }
  return false;
}


//--------------------

// Define ndarray parameters or throw message that can not do that.
bool
defineNDArrPars(PSEvt::Event& evt, const PSEvt::Source& src, const std::string& key, NDArrPars* ndarr_pars, bool print_wng)
{
  if ( defineNDArrParsForType<double>  (evt, src, key, DOUBLE,   ndarr_pars) ) return true;
  if ( defineNDArrParsForType<float>   (evt, src, key, FLOAT,    ndarr_pars) ) return true;
  if ( defineNDArrParsForType<int>     (evt, src, key, INT,      ndarr_pars) ) return true;
  if ( defineNDArrParsForType<int32_t> (evt, src, key, INT32,    ndarr_pars) ) return true;
  if ( defineNDArrParsForType<uint32_t>(evt, src, key, UINT32,   ndarr_pars) ) return true;
  if ( defineNDArrParsForType<uint16_t>(evt, src, key, UINT16,   ndarr_pars) ) return true;
  if ( defineNDArrParsForType<uint8_t> (evt, src, key, UINT8,    ndarr_pars) ) return true;
  if ( defineNDArrParsForType<int16_t> (evt, src, key, INT16,    ndarr_pars) ) return true;
  if ( defineNDArrParsForType<short>   (evt, src, key, SHORT,    ndarr_pars) ) return true;
  if ( defineNDArrParsForType<unsigned>(evt, src, key, UNSIGNED, ndarr_pars) ) return true;

  static long counter = 0; counter ++;

  if (print_wng && counter < 11) {
    const std::string msg = "ndarray shape is tested for double, uint16_t, int, float, uint8_t, int16_t, short and is not defined in the event(...)\nfor source:" 
                        + boost::lexical_cast<std::string>(src) + " key:" + key;
    MsgLog("GlobalMethods::defineNDArrPars", warning, msg);
    if (counter == 10)
       MsgLog("GlobalMethods::defineNDArrPars", warning, "STOP PRINT WARNINGS for source:" 
              << boost::lexical_cast<std::string>(src) << " key:" << key);
    //throw std::runtime_error("EXIT psana...");
  }

  return false;
}

//--------------------

void 
saveTextInFile(const std::string& fname, const std::string& text, bool print_msg)
{
  std::ofstream out(fname.c_str()); 
  out << text;
  out.close();
  //std::setprecision(9); // << std::setw(8) << std::setprecision(0) << std::fixed 

  if( print_msg ) MsgLog("GlobalMethods", info, "Save text in file " << fname.c_str());
}

//--------------------

std::string  
stringInstrument(PSEnv::Env& env)
{
  return env.instrument();
}

//--------------------

std::string  
stringExperiment(PSEnv::Env& env)
{
  return env.experiment();
}

//--------------------

unsigned 
expNum(PSEnv::Env& env)
{
  return env.expNum();
}

//--------------------

std::string
stringExpNum(PSEnv::Env& env, unsigned width)
{
  return stringFromUint(env.expNum(), width);
}

//--------------------

bool 
file_exists(std::string& fname)
{ 
  std::ifstream f(fname.c_str()); 
  return f; 
}

//--------------------
// convert source address to string
std::string srcToString( const Pds::Src& src )
{
  if ( src.level() != Pds::Level::Source ) {
    throw PSCalib::NotDetInfoError(ERR_LOC);
  }

  const Pds::DetInfo& info = static_cast<const Pds::DetInfo&>( src ) ;
  std::ostringstream str ;
  str << Pds::DetInfo::name(info.detector()) << '.' << info.detId()
      << ':' << Pds::DetInfo::name(info.device()) << '.' << info.devId() ;
  return str.str() ;
}

//--------------------

DETECTOR_TYPE detectorTypeForStringSource(const std::string& str_src)
{ 
  //std::cout << "str_src:" << str_src << '\n';
  // USE LONG NAMES FIRST, othervise detector may be misidentified!
  if      ( str_src.find("Cspad2x2")  != std::string::npos ) return CSPAD2X2;// from GlobalMethods.h
  else if ( str_src.find("Cspad")     != std::string::npos ) return CSPAD; 
  else if ( str_src.find("pnCCD")     != std::string::npos ) return PNCCD;
  else if ( str_src.find("Princeton") != std::string::npos ) return PRINCETON;
  else if ( str_src.find("Acqiris")   != std::string::npos ) return ACQIRIS;
  else if ( str_src.find("Tm6740")    != std::string::npos ) return TM6740;
  else if ( str_src.find("Opal1000")  != std::string::npos ) return OPAL1000;
  else if ( str_src.find("Opal2000")  != std::string::npos ) return OPAL2000;
  else if ( str_src.find("Opal4000")  != std::string::npos ) return OPAL4000;
  else if ( str_src.find("Opal8000")  != std::string::npos ) return OPAL8000;
  else if ( str_src.find("Andor")     != std::string::npos ) return ANDOR;
  else if ( str_src.find("OrcaFl40")  != std::string::npos ) return ORCAFL40;
  else if ( str_src.find("Fccd960")   != std::string::npos ) return FCCD960;
  else if ( str_src.find("Epix100a")  != std::string::npos ) return EPIX100A;
  else if ( str_src.find("Epix10k")   != std::string::npos ) return EPIX10K;
  else if ( str_src.find("Epix")      != std::string::npos ) return EPIX;
  else                                                       return OTHER;
}

//--------------------

DETECTOR_TYPE detectorTypeForSource(const PSEvt::Source& src)
{ 
  std::stringstream ss; ss << src;
  return detectorTypeForStringSource(ss.str());
}

//--------------------

std::string calibGroupForDetType(const DETECTOR_TYPE det_type)
{ 
  if      ( det_type == CSPAD     ) return "CsPad::CalibV1";
  else if ( det_type == CSPAD2X2  ) return "CsPad2x2::CalibV1";
  else if ( det_type == PNCCD     ) return "PNCCD::CalibV1";
  else if ( det_type == PRINCETON ) return "Princeton::CalibV1";
  else if ( det_type == ACQIRIS   ) return "Acqiris::CalibV1";
  else if ( det_type == TM6740    ) return "Camera::CalibV1";
  else if ( det_type == OPAL1000  ) return "Camera::CalibV1";
  else if ( det_type == OPAL2000  ) return "Camera::CalibV1";
  else if ( det_type == OPAL4000  ) return "Camera::CalibV1";
  else if ( det_type == OPAL8000  ) return "Camera::CalibV1";
  else if ( det_type == ANDOR     ) return "Andor::CalibV1";
  else if ( det_type == ORCAFL40  ) return "Camera::CalibV1";
  else if ( det_type == FCCD960   ) return "Camera::CalibV1";
  else if ( det_type == EPIX      ) return "Epix::CalibV1";
  else if ( det_type == EPIX100A  ) return "Epix100a::CalibV1";
  else if ( det_type == EPIX10K   ) return "Epix10k::CalibV1";
  else                              return std::string(); 
}

//--------------------

std::string stringForDetType(const DETECTOR_TYPE det_type)
{ 
  if      ( det_type == CSPAD     ) return "CSPAD";
  else if ( det_type == CSPAD2X2  ) return "CSPAD2x2";
  else if ( det_type == PNCCD     ) return "PNCCD";
  else if ( det_type == PRINCETON ) return "Princeton";
  else if ( det_type == ACQIRIS   ) return "Acqiris";
  else if ( det_type == TM6740    ) return "Camera-TM6740";
  else if ( det_type == OPAL1000  ) return "Camera-OPAL1000";
  else if ( det_type == OPAL2000  ) return "Camera-OPAL2000";
  else if ( det_type == OPAL4000  ) return "Camera-OPAL4000";
  else if ( det_type == OPAL8000  ) return "Camera-OPAL5000";
  else if ( det_type == ANDOR     ) return "Andor";
  else if ( det_type == ORCAFL40  ) return "Camera-ORCAFL40";
  else if ( det_type == FCCD960   ) return "Camera-FCCD960";
  else if ( det_type == EPIX      ) return "Epix";
  else if ( det_type == EPIX100A  ) return "Epix100a";
  else if ( det_type == EPIX10K   ) return "Epix10k";
  else                              return std::string(); 
}

//--------------------

std::string calibGroupForSource(PSEvt::Source& src)
{
    DETECTOR_TYPE det_type = detectorTypeForSource(src);
    return calibGroupForDetType(det_type);
}

//--------------------

std::string
split_string_left(const std::string& s, size_t& pos, const char& sep)
{
  size_t p0 = pos;
  size_t p1 = s.find(sep, p0);
  size_t nchars = p1-p0; 
  pos = p1+1; // move position to the next character after separator

  if (p1 != std::string::npos) return std::string(s,p0,nchars);
  else if (p0 < s.size())      return std::string(s,p0);
  else                         return std::string();
}

//--------------------

std::string strDataType(const DATA_TYPE& dtype)
{
  if      ( dtype == DOUBLE  ) return "DOUBLE";
  else if ( dtype == FLOAT   ) return "FLOAT";
  else if ( dtype == INT     ) return "INT";
  else if ( dtype == INT32   ) return "INT32";
  else if ( dtype == UINT32  ) return "UINT32";
  else if ( dtype == UINT16  ) return "UINT16";
  else if ( dtype == UINT8   ) return "UINT8";
  else if ( dtype == INT16   ) return "INT16";
  else if ( dtype == SHORT   ) return "SHORT";
  else if ( dtype == UNSIGNED) return "UNSIGNED";
  else                         return "NONDEFDT";
}

//--------------------
//--------------------
//--------------------
//--------------------
} // namespace ImgAlgos
