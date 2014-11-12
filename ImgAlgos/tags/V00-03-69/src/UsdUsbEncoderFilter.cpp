//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class UsdUsbEncoderFilter...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/UsdUsbEncoderFilter.h"
#include "ImgAlgos/GlobalMethods.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <fstream>
#include <sstream>   // for stringstream
#include <time.h>

//#include <stdlib.h> // for atoi
//#include <cstring>  // for memcpy

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"
#include "psddl_psana/usdusb.ddl.h"
#include "psddl_psana/evr.ddl.h"

//#include "PSTime/Time.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace std;
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(UsdUsbEncoderFilter)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------

UsdUsbEncoderFilter::UsdUsbEncoderFilter (const std::string& name)
  : Module(name)
  , m_source()
  , m_mode()
  , m_ifname(std::string())
  , m_ofname(std::string())
  , m_bitmask(077)
  , m_print_bits()
  , m_count_evt(0)
  , m_selected(0)
{
  m_source     = configSrc("source", "DetInfo(:USDUSB)"); 
  m_mode       = config   ("mode",           1);
  m_ifname     = configStr("ifname",        "");
  m_ofname     = configStr("ofname",        "");
  m_bitmask    = config   ("bitmask",       63);
  m_print_bits = config   ("print_bits",     0);

  loadFile();
}

//--------------------

UsdUsbEncoderFilter::~UsdUsbEncoderFilter ()
{
}

//--------------------

/// Method which is called once at the beginning of the job
void 
UsdUsbEncoderFilter::beginJob(Event& evt, Env& env)
{
  if(m_print_bits & 1) printInputParameters();printData(evt);
  if(m_print_bits & 2) printTimeCodeVector();
}

//--------------------

/// Method which is called at the beginning of the run
void 
UsdUsbEncoderFilter::beginRun(Event& evt, Env& env)
{
  if(! m_ofname.empty()) {
    if (m_print_bits & 4) MsgLog(name(), info, "Open output file: " << m_ofname);
    m_out = new std::ofstream(m_ofname.c_str()); // , std::ios::out|std::ios::app);
    if (not m_out->good()) { MsgLogRoot(error, "Failed to open output file: " << m_ofname); return; }
  }
}

//--------------------

/// Method which is called at the beginning of the calibration cycle
void 
UsdUsbEncoderFilter::beginCalibCycle(Event& evt, Env& env)
{
  if( m_mode && (m_print_bits & 16) ) printConfig(env);
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
UsdUsbEncoderFilter::event(Event& evt, Env& env)
{
  m_count_evt ++;

  if (m_print_bits & 32) printData(evt);

  if (! m_mode) { ++ m_selected; return; } // If the filter is OFF then event is selected
  

  if (! eventIsSelected(evt,env)) { skip(); return; } // event is discarded

  ++ m_selected; return; // event is selected
}
  
//--------------------

/// Method which is called at the end of the calibration cycle
void 
UsdUsbEncoderFilter::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the end of the run
void 
UsdUsbEncoderFilter::endRun(Event& evt, Env& env)
{
  if(! m_ofname.empty()) {
    if( m_print_bits & 4) MsgLog(name(), info, "Close output file " << m_ofname);
    m_out->close();
  }
}

//--------------------

/// Method which is called once at the end of the job
void 
UsdUsbEncoderFilter::endJob(Event& evt, Env& env)
{
  if( !m_mode ) return;
  if( m_print_bits & 8) MsgLog(name(), info, "Number of selected events = " << m_selected << " of total " << m_count_evt);
}

//--------------------
//--------------------
//--------------------
//--------------------
//--------------------

/// Print input parameters
void 
UsdUsbEncoderFilter::printInputParameters()
{
  std::stringstream ss; 
  ss  << "\n Input parameters:"
      << "\n m_source          : " << m_source   
      << "\n m_mode            : " << m_mode   
      << "\n m_ifname          : " << m_ifname 
      << "\n m_ofname          : " << m_ofname 
      << "\n m_bitmask(oct)    : " << std::oct << int(m_bitmask)
      << "\n m_print_bits(oct) : " << std::oct << m_print_bits
      << "\n";

  MsgLog(name(), info, ss.str());
}

//--------------------

/// Print input parameters
void 
UsdUsbEncoderFilter::printTimeCodeVector()
{
  std::stringstream ss; 
  ss << "\nContent of the input file in vector<TimeCode> of size: " << v_tcode.size() << '\n';
  ss << "#    t-stamp sec     nsec    code   evt# \n";
  unsigned counter(0);
  std::vector<TimeCode>::const_iterator it;
  for(it=v_tcode.begin(); it!=v_tcode.end(); it++) {
    ss << right << std::setw(6) << ++counter << ":  " << *it << '\n';
    if (counter > 8) {ss << "    ...\n"; break;}
  }

  MsgLog(name(), info, ss.str());
}

//--------------------

void
UsdUsbEncoderFilter::loadFile()
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
  tstamp_t ts_sec; 
  tstamp_t ts_nsec; 
  unsigned ucode;
  evnum_t  evnum;
  while(in >> ts_sec && in >> ts_nsec && in >> ucode && in >> evnum) {
    code_t code(ucode);
    v_tcode.push_back(TimeCode(ts_sec, ts_nsec, code, evnum));
  }

  /*
  TimeCode tc();
  while(in >> tc) {
    v_tcode.push_back(tc);
  }
  */

  // close file
  in.close();

  // initialize iterator pointing to the beginning of the vector
  v_tc_iter = v_tcode.begin(); 
}

//--------------------

bool
UsdUsbEncoderFilter::eventIsSelected(Event& evt, Env& env)
{
  TimeCode tc_data; 

  //MsgLog(name(), info, "event: " << m_count_evt);
  cout  << "\n====Event #" << m_count_evt << "   Precise current time: " << str_current_time() << '\n';


  // --- Get timestamp from EventId

  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    cout << "  Time from EventId: "  << eventId->time().sec() << "(sec) "
         << eventId->time().nsec() << "(nsec) "
         << "or the same in double: "  << fixed << std::setw(20) << std::setprecision(9) << doubleTime(evt)
         << " or as date-time: " << stringTimeStamp(evt)
         << '\n';

    // Update the timestamp using eventId
    tc_data.set_tst_sec(eventId->time().sec());
    tc_data.set_tst_nsec(eventId->time().nsec());
    tc_data.set_evnum(m_count_evt);
  }


  // --- Get code from UsdUsb

  shared_ptr<Psana::UsdUsb::DataV1> data1 = evt.get(m_source);
  if (data1) {
    //tstamp_t tstamp = data1->timestamp();
    code_t code = data1->digital_in() & m_bitmask;
    tc_data.set_code(code);
  }

  cout << "  TimeCode in data: " << tc_data << '\n';


  // --- Get timestamp from Evr
  // Source src_evr("DetInfo(:Evr)");
  // shared_ptr<Psana::EvrData::DataV3> data3 = evt.get(src_evr);
  // if (data3) {
  //   cout << "  EvrData::DataV3: numFifoEvents=" << data3->numFifoEvents();
  //   const ndarray<const Psana::EvrData::FIFOEvent, 1>& array = data3->fifoEvents();
  //   for (unsigned i = 0; i < array.size(); ++ i) {
  // 
  //     cout  << "\n    fifo event #" << i
  //           <<  " timestampHigh=" << array[i].timestampHigh()
  //           <<  " timestampLow=" << array[i].timestampLow()
  //           <<  " eventCode=" << array[i].eventCode()
  //           << "\n";
  //   } 
  //   // Update the timestamp using Evr
  //   //tc_data.set_tst_sec(array[0].timestampHigh())
  // }


  // --- Add record to the output file, if needed (if output file name parameter "ofname" is specified)
  if(! m_ofname.empty() && m_out->good()) *m_out << tc_data << '\n'; 


  // --- 
  // At this stage tc_data object should be completely defined from data.
  // Next step: tc_data object should be compared with content of the input file
  // stored in std::vector<TimeCode> v_tcode;
  // return true/false if the event should be selected/discarded

  // OPERATORS "<" and "==" SHOULD BE PROPERLY DEFINED IN UsdUsbEncoderFilter.h

  // --- iterate over the input records (stored in v_tcode) and find the best match

  while ( *v_tc_iter < tc_data && v_tc_iter != v_tcode.end() ) ++v_tc_iter;

  //===== Selector decision ============

  if ( *v_tc_iter == tc_data ) return (m_mode > 0) ? true  : false;
  else                         return (m_mode > 0) ? false : true;

  //return true;  // if event is selected
  //return false; // if event is discarded
  //====================================

}

//--------------------

void 
UsdUsbEncoderFilter::printConfig(Env& env)
{
  shared_ptr<Psana::UsdUsb::ConfigV1> config1 = env.configStore().get(m_source);
  if (config1) {
    WithMsgLog(name(), info, str) {
      str << "UsdUsb::ConfigV1:";
      str << "\n  counting_mode = " << config1->counting_mode();
      str << "\n  quadrature_mode = " << config1->quadrature_mode();
    }
  }
}

//--------------------

void 
UsdUsbEncoderFilter::printData(Event& evt)
{
  shared_ptr<Psana::UsdUsb::DataV1> data1 = evt.get(m_source);
  if (data1) {
    WithMsgLog(name(), info, str) {
      str << "UsdUsb::DataV1:";
      str << "\n  encoder_count = " << data1->encoder_count();
      str << "\n  analog_in = " << data1->analog_in();
      ndarray<const uint8_t, 1> st = data1->status();
      str << "\n  status = [" << int(st[0]) << ' ' << int(st[1]) << ' ' << int(st[2]) << ' ' << int(st[3]) <<']' ;
      str << "\n  digital_in = " << int(data1->digital_in());
      str << "\n  timestamp = " << int(data1->timestamp());
    }
  }
}

//--------------------

std::string 
UsdUsbEncoderFilter::str_current_time()
{
  struct timespec ctime;
  // int gettimeStatus = 
  clock_gettime( CLOCK_REALTIME, &ctime );
  std::stringstream ss; 
  ss << ctime.tv_sec << "(sec) " << ctime.tv_nsec << "(nsec)";
  return ss.str();
}

//--------------------
//--------------------
//--------------------
//--------------------
} // namespace ImgAlgos
