//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgSaveInFile...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/ImgSaveInFile.h"

//-----------------
// C/C++ Headers --
//-----------------
// #include <time.h>
#include <sstream> // for stringstream

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
//#include <boost/lexical_cast.hpp>
#include <iomanip> // for setw, setfill
#include <sstream> // for stringstream

// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(ImgSaveInFile)

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------

ImgSaveInFile::ImgSaveInFile (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key()
  , m_eventSave()
  , m_saveAll()
  , m_fname()
  , m_file_type()
  , m_print_bits()
  , m_count(0)
  , m_count_msg(0)
{
  // get the values from configuration or use defaults
  m_str_src       = configSrc("source",   "CxiDs1.0:Cspad.0");
  m_key           = configStr("key",      "Image2D");
  m_eventSave     = config   ("eventSave", 0);
  m_saveAll       = config   ("saveAll",   false);
  m_fname         = configStr("fname",    "img");
  m_file_type     = configStr("ftype",    "txt");
  m_print_bits    = config   ("print_bits",0);

  setFileMode();
}

//--------------------
/// Print input parameters
void 
ImgSaveInFile::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\nInput parameters:"
        << "\nsource       : "     << m_str_src
        << "\nkey          : "     << m_key      
        << "\neventSave    : "     << m_eventSave
        << "\nsaveAll      : "     << m_saveAll
        << "\nfname        : "     << m_fname
        << "\nftype        : "     << m_file_type
        << "\nm_print_bits : "     << m_print_bits
        << "\n";
  }
}

//--------------------

void 
ImgSaveInFile::setFileMode()
{
  m_file_mode = TEXT;
  if (m_file_type == "bin")  { m_file_mode = BINARY; return; }
  if (m_file_type == "txt")  { m_file_mode = TEXT;   return; }
  if (m_file_type == "tiff") { m_file_mode = TIFF;   return; }
  if (m_file_type == "png")  { m_file_mode = PNG;    return; }

  const std::string msg = "The output file type: " + m_file_type + " is not recognized. Known types are: bin, txt";
  MsgLogRoot(error, msg);
  throw std::runtime_error(msg);
}


//--------------
// Destructor --
//--------------

ImgSaveInFile::~ImgSaveInFile ()
{
}

//--------------------

//--------------------

/// Method which is called once at the beginning of the job
void 
ImgSaveInFile::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
}

//--------------------

/// Method which is called at the beginning of the run
void 
ImgSaveInFile::beginRun(Event& evt, Env& env)
{
  m_str_runnum     = stringRunNumber(evt);
  m_str_experiment = stringExperiment(env);
  m_fname_common   = m_fname + "-" + m_str_experiment + "-r" + m_str_runnum;
}

//--------------------

/// Method which is called at the beginning of the calibration cycle
void 
ImgSaveInFile::beginCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ImgSaveInFile::event(Event& evt, Env& env)
{
  // this is how to gracefully stop analysis job
  ++m_count;
  //if (m_count >= m_maxEvents) stop();
  MsgLog(name(), debug, "Event: " << m_count);
  if (m_saveAll || m_count == m_eventSave) this -> saveImageInFile(evt);
}

//--------------------
  
/// Method which is called at the end of the calibration cycle
void 
ImgSaveInFile::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the end of the run
void 
ImgSaveInFile::endRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called once at the end of the job
void 
ImgSaveInFile::endJob(Event& evt, Env& env)
{
}

//--------------------
//--------------------
//--------------------
//--------------------

void 
ImgSaveInFile::saveImageInFile(Event& evt)
{
  // Define the file name
  string fname = m_fname_common
        + "-e"    + stringFromUint(m_count,8,'0')
        + "-"     + stringTimeStamp(evt) 
        + "."     + m_file_type;
      //+ "-ev"   + stringFromUint(m_count)
      //+ ".txt";

  if ( save2DArrayInFileForType <float>           (evt, m_str_src, m_key, fname, m_print_bits & 2, m_file_mode) ) return;
  if ( save2DArrayInFileForType <int16_t>         (evt, m_str_src, m_key, fname, m_print_bits & 2, m_file_mode) ) return;
  if ( save2DArrayInFileForType <double>          (evt, m_str_src, m_key, fname, m_print_bits & 2, m_file_mode) ) return;
  if ( save2DArrayInFileForType <uint16_t>        (evt, m_str_src, m_key, fname, m_print_bits & 2, m_file_mode) ) return;
  if ( save2DArrayInFileForType <int>             (evt, m_str_src, m_key, fname, m_print_bits & 2, m_file_mode) ) return;
  if ( save2DArrayInFileForType <uint8_t>         (evt, m_str_src, m_key, fname, m_print_bits & 2, m_file_mode) ) return;
  if ( save2DArrayInFileForType <unsigned short>  (evt, m_str_src, m_key, fname, m_print_bits & 2, m_file_mode) ) return;
       
  if ( saveImage2DInFileForType <float>           (evt, m_str_src, m_key, fname, m_print_bits & 2) ) return;
  if ( saveImage2DInFileForType <int16_t>         (evt, m_str_src, m_key, fname, m_print_bits & 2) ) return;
  if ( saveImage2DInFileForType <double>          (evt, m_str_src, m_key, fname, m_print_bits & 2) ) return;
  if ( saveImage2DInFileForType <uint16_t>        (evt, m_str_src, m_key, fname, m_print_bits & 2) ) return;
  if ( saveImage2DInFileForType <int>             (evt, m_str_src, m_key, fname, m_print_bits & 2) ) return;
  if ( saveImage2DInFileForType <uint8_t>         (evt, m_str_src, m_key, fname, m_print_bits & 2) ) return;
  if ( saveImage2DInFileForType <unsigned short>  (evt, m_str_src, m_key, fname, m_print_bits & 2) ) return;

  if (++m_count_msg < 11 && m_print_bits) {
    std::stringstream ss; ss << "Image is not defined in the event(...) for source:" << m_str_src << " key:" << m_key;
    MsgLogRoot(warning, ss.str());
    if (m_count_msg == 10) MsgLog(name(), warning, "STOP WARNINGS for source:" << m_str_src << " key:" << m_key);    
  //throw std::runtime_error(msg);
  }
}

//--------------------
//--------------------

} // namespace ImgAlgos
