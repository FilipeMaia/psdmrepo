//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EventCodeFilter...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/EventCodeFilter.h"
#include "ImgAlgos/GlobalMethods.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>  // for stringstream
#include <iostream> // for cout
#include <stdint.h> // uint8_t, uint32_t, etc.

//#include <time.h>
//#include <fstream>
//#include <stdlib.h> // for atoi
//#include <cstring>  // for memcpy
//#include <iomanip>  // for setw, setfill

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/evr.ddl.h"
#include "PSEvt/Source.h"

//#include "PSEvt/EventId.h"
//#include "PSTime/Time.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace std;
using namespace ImgAlgos;

PSANA_MODULE_FACTORY(EventCodeFilter)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace {

// Helper methods to print individual data objects

//--------------------

  void print(std::ostream& str, unsigned i, const Psana::EvrData::FIFOEvent& f)
  {
    str << "\n  fifo event #" << i
        << " timestampHigh=" << f.timestampHigh()
        << " timestampLow=" << f.timestampLow()
        << " eventCode="   << f.eventCode();
  }

//--------------------

  template <typename T>
  void print_array(std::ostream& str, const ndarray<T, 1>& array) {
    for (unsigned i = 0; i < array.size(); ++ i) {
      ::print(str, i, array[i]);
    }
  }

//--------------------

  template <typename T> // T = Psana::EvrData::DataV3, 4, ...
  bool printFIFOEventsInEventForType(Event& evt, const Source& source, const unsigned& vers)
  {
    shared_ptr<T> data = evt.get(source);
    if (data) {    
      WithMsgLog("EventCodeFilter::printFIFOEvents...", info, str) {
        str << "EvrData::DataV" << vers << ": numFifoEvents=" << data->numFifoEvents();
        ::print_array(str, data->fifoEvents());
      }
      return true;
    }
    return false;
  }

//--------------------

  template <typename T> // T = Psana::EvrData::DataV3, 4, ...
  bool evcodeIsAvailableForType(Event& evt, const Source& source, const uint32_t& evcode) {

    shared_ptr<T> data = evt.get(source);
    if (data) {
      //std::cout << "EvrData::DataV3, 4,...: numFifoEvents=" << data->numFifoEvents();
      const ndarray<const Psana::EvrData::FIFOEvent, 1>& array = data->fifoEvents();

      for (unsigned i = 0; i < array.size(); ++ i) {
	//std::cout << " evcode="   << array[i].eventCode();;

	if(array[i].eventCode() == evcode) return true;
      }
    }
    return false;
  }

//--------------------
} // namespace 


//--------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------

EventCodeFilter::EventCodeFilter (const std::string& name)
  : Module(name)
  , m_source()
  , m_evcode()
  , m_mode()
  , m_print_bits()
  , m_count_evt(0)
  , m_count_sel(0)
//, m_str_evcodes()
{
  m_source      = configSrc("source", "DetInfo(:Evr)");
  m_evcode      = config   ("evtcode",        0);
  m_mode        = config   ("mode",           0);
  m_print_bits  = config   ("print_bits",     0);
  //m_str_evcodes = configStr("evtcode",     "");

  //convertStringEvtCodesToVector();
}

//--------------------

EventCodeFilter::~EventCodeFilter ()
{
}

//--------------------

/// Method which is called once at the beginning of the job
void 
EventCodeFilter::beginJob(Event& evt, Env& env)
{
  if(m_print_bits & 1) printInputParameters();
  //  if(m_print_bits & 2) printEventCodeVector();
}

//--------------------

/// Method which is called at the beginning of the run
void 
EventCodeFilter::beginRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the beginning of the calibration cycle
void 
EventCodeFilter::beginCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
EventCodeFilter::event(Event& evt, Env& env)
{
  m_count_evt ++;
  bool is_selected(true);

  if (! m_mode) is_selected = true;  // If the filter is OFF then event is selected
  else if (! eventIsSelected(evt,env)) { skip(); is_selected = false; } // event is discarded

  if (m_print_bits & 8) MsgLog(name(), info, "Event #" << m_count_evt
			       << ((is_selected) ? " is selected " :  " is discarded")
                               << " for event code " << m_evcode);
  if (is_selected) ++m_count_sel;

  if (m_print_bits & 16) printFIFOEventsInEvent(evt);
}
  
//--------------------

/// Method which is called at the end of the calibration cycle
void 
EventCodeFilter::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the end of the run
void 
EventCodeFilter::endRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called once at the end of the job
void 
EventCodeFilter::endJob(Event& evt, Env& env)
{
  if(! m_mode) return;
  if(m_print_bits & 4) MsgLog(name(), info, "Number of selected events = " << m_count_sel << " of total " << m_count_evt);
}

//--------------------
//--------------------
//--------------------
//--------------------
//--------------------

/// Print input parameters
void 
EventCodeFilter::printInputParameters()
{
  std::stringstream ss; 
  ss  << "\n Input parameters:"
      << "\n m_source         : " << m_source
      << "\n m_evcode         : " << m_evcode
      << "\n m_mode           : " << m_mode   
      << "\n m_print_bits dec : " << m_print_bits
      << "  oct : "   << std::oct << m_print_bits
      << "  hex : "   << std::hex << m_print_bits
      << "\n";
  //<< "\n m_str_evcodes     : " << m_str_evcodes

  MsgLog(name(), info, ss.str());
}

//--------------------

//void
//EventCodeFilter::convertStringEvtCodesToVector()
//{
//  if (m_str_evcodes.empty()) { MsgLogRoot(error, "Input string of event codes is empty"); return; }
//  std::stringstream ss; ss << m_str_evcodes;
//  int evcode; 
//  while(ss >> evcode) v_evcode.push_back(evcode);
//}

//--------------------

/// Print input parameters
//void 
//EventCodeFilter::printEventCodeVector()
//{
//  std::stringstream ss; 
//  ss << "\nInput array of event numbers of size: " << v_evcode.size() << '\n';
//  unsigned counter(0);
//  std::vector<int>::const_iterator it;
//  for(it=v_evcode.begin(); it!=v_evcode.end(); it++) {
//    ss << right << std::setw(6) << " " << *it;
//    if (counter > 9) {ss << '\n'; counter=0;}
//  }
//  MsgLog(name(), info, ss.str());
//}

//--------------------

bool
EventCodeFilter::eventIsSelected(Event& evt, Env& env)
{
  bool status = evcodeIsAvailable(evt, env);

  // selector decision
  if (status) return (m_mode > 0) ? true  : false;
  else        return (m_mode > 0) ? false : true;

  //return true;  // if event is selected
  //return false; // if event is discarded
  //====================================
}

//--------------------

bool
EventCodeFilter::evcodeIsAvailable(Event& evt, Env& env)
{
  if      (::evcodeIsAvailableForType<Psana::EvrData::DataV4>(evt, m_source, m_evcode)) return true;
  else if (::evcodeIsAvailableForType<Psana::EvrData::DataV3>(evt, m_source, m_evcode)) return true;
  else return false;  
}

//--------------------

void 
EventCodeFilter::printFIFOEventsInEvent(Event& evt)
{
  if      (::printFIFOEventsInEventForType<Psana::EvrData::DataV4>(evt, m_source, 4)) return;
  else if (::printFIFOEventsInEventForType<Psana::EvrData::DataV3>(evt, m_source, 3)) return;
  else return;  
}

//--------------------
//--------------------
//--------------------
//--------------------
} // namespace ImgAlgos
