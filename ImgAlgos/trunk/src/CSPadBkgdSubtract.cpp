//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadBkgdSubtract...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/CSPadBkgdSubtract.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
// to work with detector data include corresponding 
// header from psddl_psana package
#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(CSPadBkgdSubtract)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
CSPadBkgdSubtract::CSPadBkgdSubtract (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_str_src    = configStr("source", "DetInfo(:Cspad)");
  m_key        = configStr("key", "");
  m_fname      = configStr("bkgd_fname", "cspad_background.dat");
  m_print_bits = config("print_bits", 0);
}

//--------------
// Destructor --
//--------------
CSPadBkgdSubtract::~CSPadBkgdSubtract ()
{
}

/// Method which is called once at the beginning of the job
void 
CSPadBkgdSubtract::beginJob(Event& evt, Env& env)
{
  getBkgdArray();
  printBkgdArray();
  if( m_print_bits & 1 ) printInputParameters();
}

/// Method which is called at the beginning of the run
void 
CSPadBkgdSubtract::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
CSPadBkgdSubtract::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPadBkgdSubtract::event(Event& evt, Env& env)
{
  ++ m_count;

  if( m_print_bits & 2 ) printEventId(evt);

}
  
/// Method which is called at the end of the calibration cycle
void 
CSPadBkgdSubtract::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
CSPadBkgdSubtract::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
CSPadBkgdSubtract::endJob(Event& evt, Env& env)
{
}

//--------------------

// Print input parameters
void 
CSPadBkgdSubtract::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n source     : " << m_str_src
        << "\n key        : " << m_key      
        << "\n m_fname    : " << m_fname    
        << "\n print_bits : " << m_print_bits
        << "\n";     

    log << "\n MaxQuads   : " << MaxQuads    
        << "\n MaxSectors : " << MaxSectors  
        << "\n NumColumns : " << NumColumns  
        << "\n NumRows    : " << NumRows     
        << "\n SectorSize : " << SectorSize  
        << "\n";
  }
}

//--------------------

void 
CSPadBkgdSubtract::printEventId(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    //MsgLog( name(), info, "Event="  << m_count << " ID: " << *eventId);
    MsgLog( name(), info, "Event="  << m_count << " time: " << stringTimeStamp(evt) );
  }
}

//--------------------

std::string
CSPadBkgdSubtract::stringTimeStamp(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    return (eventId->time()).asStringFormat("%Y%m%dT%H:%M:%S%f"); //("%Y-%m-%d %H:%M:%S%f%z");
  }
  return std::string("Time-stamp-is-unavailable");
}

//--------------------

void
CSPadBkgdSubtract::getBkgdArray()
{
  MsgLog( name(), info, "::getBkgdArray()" );
  openFile();
  readArrFromFile();
  closeFile();
}

//--------------------

void
CSPadBkgdSubtract::openFile()
{
   m_file.open(m_fname.c_str());
}

//--------------------

void
CSPadBkgdSubtract::closeFile()
{
   m_file.close();
}

//--------------------

void 
CSPadBkgdSubtract::readArrFromFile()
{
  v_parameters.clear();
  std::string str;
  do{ 
      m_file >> str; 
      if(m_file.good()) {
         v_parameters.push_back(std::atof(str.c_str())); // cout << str << " "; 
      }
    } while( m_file.good() );                            // cout << endl << endl;
}

//--------------------

void 
CSPadBkgdSubtract::printBkgdArray()
{
  for( std::vector<double>::const_iterator itv  = v_parameters.begin();
                                           itv != v_parameters.end(); itv++ ) {
    cout << *itv << ' ';
  }
}

//--------------------


} // namespace ImgAlgos
