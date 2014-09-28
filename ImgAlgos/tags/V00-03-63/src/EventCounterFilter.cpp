//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EventCounterFilter...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/EventCounterFilter.h"
#include "ImgAlgos/GlobalMethods.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <fstream>
#include <sstream>   // for stringstream
#include <time.h>
#include <iostream> // for cout

//#include <stdlib.h> // for atoi
//#include <cstring>  // for memcpy

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"

//#include "PSTime/Time.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace std;
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(EventCounterFilter)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------

EventCounterFilter::EventCounterFilter (const std::string& name)
  : Module(name)
  , m_mode()
  , m_ifname(std::string())
  , m_print_bits()
  , m_count_evt(0)
  , m_count_sel(0)
{
  m_mode       = config   ("mode",           1);
  m_ifname     = configStr("ifname",        "");
  m_print_bits = config   ("print_bits",     0);

  loadFile();
}

//--------------------

EventCounterFilter::~EventCounterFilter ()
{
}

//--------------------

/// Method which is called once at the beginning of the job
void 
EventCounterFilter::beginJob(Event& evt, Env& env)
{
  if(m_print_bits & 1) printInputParameters();
  if(m_print_bits & 2) printEventVector();
}

//--------------------

/// Method which is called at the beginning of the run
void 
EventCounterFilter::beginRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the beginning of the calibration cycle
void 
EventCounterFilter::beginCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
EventCounterFilter::event(Event& evt, Env& env)
{
  m_count_evt ++;
  if( m_print_bits & 16) cout  << "\nevent #" << m_count_evt;

  if (! m_mode) { ++ m_count_sel; return; } // If the filter is OFF then event is selected
  
  if (! eventIsSelected(evt,env)) { skip(); return; } // event is discarded

  if( m_print_bits & 32) cout << " is selected";

  ++ m_count_sel; 
  return; // event is selected
}
  
//--------------------

/// Method which is called at the end of the calibration cycle
void 
EventCounterFilter::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the end of the run
void 
EventCounterFilter::endRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called once at the end of the job
void 
EventCounterFilter::endJob(Event& evt, Env& env)
{
  if( !m_mode ) return;
  if( m_print_bits & 4) MsgLog(name(), info, "Number of selected events = " << m_count_sel << " of total " << m_count_evt);
}

//--------------------
//--------------------
//--------------------
//--------------------
//--------------------

/// Print input parameters
void 
EventCounterFilter::printInputParameters()
{
  std::stringstream ss; 
  ss  << "\n Input parameters:"
      << "\n m_mode            : " << m_mode   
      << "\n m_ifname          : " << m_ifname 
      << "\n m_print_bits(oct) : " << std::oct << m_print_bits
      << "\n";

  MsgLog(name(), info, ss.str());
}

//--------------------

/// Print input parameters
void 
EventCounterFilter::printEventVector()
{
  std::stringstream ss; 
  ss << "\nInput array of event numbers of size: " << v_evnum.size() << '\n';
  unsigned counter(0);
  std::vector<unsigned>::const_iterator it;
  for(it=v_evnum.begin(); it!=v_evnum.end(); it++) {
    ss << right << std::setw(6) << " " << *it;
    if (counter > 9) {ss << '\n'; counter=0;}
  }
  MsgLog(name(), info, ss.str());
}

//--------------------

void
EventCounterFilter::loadFile()
{
  if (m_ifname.empty()) { MsgLogRoot(error, "Input file name is empty"); return; }

  // open file     
  std::ifstream in(m_ifname.c_str());
  if (not in.good()) {
    const std::string msg = "Failed to open file: " + m_ifname;
    MsgLogRoot(error, msg);
    //throw std::runtime_error(msg);
    return;
  }

  // read all values and store them in vector
  unsigned evnum;
  while(in >> evnum) v_evnum.push_back(evnum);

  // close file
  in.close();

  // initialize iterator pointing to the beginning of the vector
  v_evnum_iter = v_evnum.begin(); 
}

//--------------------

bool
EventCounterFilter::eventIsSelected(Event& evt, Env& env)
{
  // scroll vector of selected event numbers
  while ( *v_evnum_iter < m_count_evt && v_evnum_iter < v_evnum.end()-1 ) ++v_evnum_iter;
  //std::cout << "    ------> m_count_evt:" << m_count_evt << " *v_evnum_iter: " << *v_evnum_iter << "\n";

  // selector decision
  if ( *v_evnum_iter == m_count_evt ) return (m_mode > 0) ? true  : false;
  else                                return (m_mode > 0) ? false : true;

  //return true;  // if event is selected
  //return false; // if event is discarded
  //====================================

}

//--------------------
//--------------------
//--------------------
//--------------------
} // namespace ImgAlgos
