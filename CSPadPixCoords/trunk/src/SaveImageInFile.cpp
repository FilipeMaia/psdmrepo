//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class SaveImageInFile...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadPixCoords/SaveImageInFile.h"

//-----------------
// C/C++ Headers --
//-----------------
// #include <time.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"
#include "CSPadPixCoords/Image2D.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
//#include <boost/lexical_cast.hpp>
#include <iomanip> // for setw, setfill
#include <sstream> // for streamstring

// This declares this class as psana module
using namespace CSPadPixCoords;
PSANA_MODULE_FACTORY(SaveImageInFile)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace CSPadPixCoords {

//----------------
// Constructors --
//----------------

SaveImageInFile::SaveImageInFile (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key()
  , m_eventSave()
  , m_saveAll()
  , m_fname()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_str_src       = configStr("source",   "CxiDs1.0:Cspad.0");
  m_key           = configStr("key",      "Image2D");
  m_eventSave     = config   ("eventSave", 0);
  m_saveAll       = config   ("saveAll",   false);
  m_fname         = configStr("fname",    "cspad-image");
  m_print_bits    = config   ("print_bits",0);
}

//--------------
// Destructor --
//--------------

SaveImageInFile::~SaveImageInFile ()
{
}

//--------------------

//--------------------

/// Method which is called once at the beginning of the job
void 
SaveImageInFile::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
}

//--------------------

/// Method which is called at the beginning of the run
void 
SaveImageInFile::beginRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the beginning of the calibration cycle
void 
SaveImageInFile::beginCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
SaveImageInFile::event(Event& evt, Env& env)
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
SaveImageInFile::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the end of the run
void 
SaveImageInFile::endRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called once at the end of the job
void 
SaveImageInFile::endJob(Event& evt, Env& env)
{
}

//--------------------
//--------------------
//--------------------
//--------------------

std::string  
SaveImageInFile::strTimeStamp(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {

    //m_time = eventId->time();
    //std::stringstream ss;
    //ss << hex << t_msec;
    //string hex_msec = ss.str();

    return (eventId->time()).asStringFormat( "%Y-%m-%d-%H%M%S%f"); // "%Y-%m-%d %H:%M:%S%f%z"
  }
  else
    return std::string("time-stamp-is-not-defined");
}

//--------------------

std::string  
SaveImageInFile::strRunNumber(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    stringstream ssRunNum; ssRunNum << "r" << setw(4) << setfill('0') << eventId->run();
    return ssRunNum.str();
  }
  else
    return std::string("run-is-not-defined");
}

//--------------------

std::string
SaveImageInFile::strEventCounter()
{
  stringstream ssEvNum; ssEvNum << setw(6) << setfill('0') << m_count;
  return ssEvNum.str();
}

//--------------------

void 
SaveImageInFile::saveImageInFile(Event& evt)
{
  // Define the file name
  stringstream ssEvNum; ssEvNum << setw(6) << setfill('0') << m_count;
  string fname = m_fname + "-" + strRunNumber(evt) + "-" + strTimeStamp(evt) + ".txt";

  // In case if m_key == "Image2D" 

  shared_ptr< CSPadPixCoords::Image2D<double> > img2d = evt.get(m_str_src, m_key, &m_src);
  if (img2d.get()) {
    if( m_print_bits & 2 )MsgLog(name(), info, "::saveImageInFile(...): Get image as Image2D<double> from event and save it in file");
    img2d -> saveImageInFile(fname,0);
  } // if (img2d.get())


  shared_ptr< ndarray<double,2> > img = evt.get(m_str_src, m_key, &m_src);
  if (img.get()) {
    if( m_print_bits & 2 ) MsgLog(name(), info, "::saveImageInFile(...): Get image as ndarray<double,2> from event and save it in file");
    CSPadPixCoords::Image2D<double> *img2d = new CSPadPixCoords::Image2D<double>(img->data(),img->shape()[0],img->shape()[1]);
    img2d -> saveImageInFile(fname,0);
  } // if (img.get())


  shared_ptr< ndarray<uint16_t,2> > img_u16 = evt.get(m_str_src, m_key, &m_src);
  if (img_u16.get()) {
    if( m_print_bits & 2 ) MsgLog(name(), info, "::saveImageInFile(...): Get image as ndarray<uint16_t,2> from event and save it in file");
    CSPadPixCoords::Image2D<uint16_t> *img2d = new CSPadPixCoords::Image2D<uint16_t>(img_u16->data(),img_u16->shape()[0],img_u16->shape()[1]);
    img2d -> saveImageInFile(fname,0);
  } // if (img_u16.get())
}

//--------------------
/// Print input parameters
void 
SaveImageInFile::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\nInput parameters:"
        << "\nsource       : "     << m_str_src
        << "\nkey          : "     << m_key      
        << "\neventSave    : "     << m_eventSave
        << "\nsaveAll      : "     << m_saveAll
        << "\nfname        : "     << m_fname
        << "\nm_print_bits : "     << m_print_bits;
  }
}

//--------------------
//--------------------

} // namespace CSPadPixCoords
