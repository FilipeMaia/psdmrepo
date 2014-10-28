//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPad2x2NDArrReshape...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadPixCoords/CSPad2x2NDArrReshape.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <time.h>
#include <sstream>   // for stringstream

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
//#include <boost/lexical_cast.hpp>

// This declares this class as psana module
using namespace CSPadPixCoords;
PSANA_MODULE_FACTORY(CSPad2x2NDArrReshape)

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace CSPadPixCoords {

//----------------
// Constructors --
//----------------

CSPad2x2NDArrReshape::CSPad2x2NDArrReshape (const std::string& name)
  : Module(name)
  , m_source()
  , m_keys_in()
  , m_keys_out()
  , m_print_bits()
  , m_count_evt(0)
  , m_count_clb(0)
  , m_count_msg(0)
{
  m_source        = configSrc("source",     ":Cspad2x2");
  m_keys_in       = configStr("keys_in",    "");
  m_keys_out      = configStr("keys_out",   "");
  m_print_bits    = config   ("print_bits", 0);

  put_keys_in_vectors();

  //m_config = new CONFIG ( m_source ); 
}

//--------------
// Destructor --
//--------------

CSPad2x2NDArrReshape::~CSPad2x2NDArrReshape ()
{
}

//--------------------

/// Print input parameters
void 
CSPad2x2NDArrReshape::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\nInput parameters:"
        << "\nsource        : "     << m_source
        << "\nkeys_in       : "     << m_keys_in
        << "\nkeys_out      : "     << m_keys_out
        << "\nprint_bits    : "     << m_print_bits
        << "\n";
  }
}

//--------------------

/// Method which is called once at the beginning of the job
void 
CSPad2x2NDArrReshape::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
  if( m_print_bits & 2 ) print_in_out_keys();
  if( m_print_bits & 16) printSizeOfTypes();
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPad2x2NDArrReshape::event(Event& evt, Env& env)
{
  ++m_count_evt;

  procEvent(evt);

  if( m_count_clb == 0 ) {
    if(procCalib(env)) m_count_clb++;
  }
}

//--------------------

void CSPad2x2NDArrReshape::beginRun(Event& evt, Env& env) 
{ 
  m_count_clb = 0; 
}

//--------------------

void CSPad2x2NDArrReshape::beginCalibCycle(Event& evt, Env& env) {}
void CSPad2x2NDArrReshape::endCalibCycle(Event& evt, Env& env) {}
void CSPad2x2NDArrReshape::endRun(Event& evt, Env& env) {}
void CSPad2x2NDArrReshape::endJob(Event& evt, Env& env) {}

//--------------------

void
CSPad2x2NDArrReshape::put_keys_in_vectors()
{
  v_keys_in     .reserve(50);
  v_keys_out    .reserve(50);

  std::stringstream ss(m_keys_in);
  std::string key_in;  
  std::string key_out;  

  if(m_keys_in.empty()) {
    MsgLog(name(), warning, "The list of input keys in keys_in is empty. " 
                         << "At lease one input key needs to be specified.");
        throw std::runtime_error("Check CSPad2x2NDArrReshape parameters in the configuration file!");
  }

  if(m_keys_out.empty()) {
    // Auto-generate output keys

      do { ss >> key_in;
        size_t pos = key_in.find(':');
        if( pos != std::string::npos ) key_out = std::string(key_in, 0, pos);
	else                           key_out = key_in + ":2x185x388";

        //cout  << " key_in: " << std::setw(20) << key_in 
        //      << " key_out:" << std::setw(20) << key_out 
        //      << '\n'; 

        v_keys_in .push_back(key_in); 
        v_keys_out.push_back(key_out); 
      
      } while( ss.good() ); 

  } else {
    // Take output keys from config file

      std::stringstream ss_out(m_keys_out);

      do { ss     >> key_in;  v_keys_in .push_back(key_in);  } while( ss.good() ); 
      do { ss_out >> key_out; v_keys_out.push_back(key_out); } while( ss_out.good() ); 

      //cout  << "Input keys  : " << m_keys_in << '\n';  
      //cout  << "Output keys : " << m_keys_out << '\n';  
      //cout  << "Number of input keys  : " << v_keys_in .size() << '\n';  
      //cout  << "Number of output keys : " << v_keys_out.size() << '\n';  

      if(v_keys_in.size() != v_keys_out.size()) {
        MsgLog(name(), warning, "The list of output keys: (" << m_keys_out
               << ") is not empty, \nbut their number: " << v_keys_out.size() 
               <<" does not equal to the number: " << v_keys_in.size() 
	       <<" of the input keys: (" << m_keys_in << ").");
        throw std::runtime_error("Check CSPad2x2NDArrReshape parameters in the configuration file!");
      }
  }
}

//--------------------

void
CSPad2x2NDArrReshape::print_in_out_keys()
{
      std::vector<std::string>::iterator it_in  = v_keys_in.begin();
      std::vector<std::string>::iterator it_out = v_keys_out.begin();

      std::stringstream ss; ss << "\n  List of input/output keys for [185x388x2] -> [2x185x388] conversion:";

      for ( ; it_in != v_keys_in.end(); ++it_in, ++it_out) 
	ss  << "\n    key_in: "  << std::setw(16) << std::left << *it_in 
            << "  key_out: "     << std::setw(16) << std::left << *it_out; 
      ss  << '\n';

      MsgLog(name(), info, ss.str());
}

//--------------------

bool
CSPad2x2NDArrReshape::procEvent(Event& evt)
{  
  bool is_found = false;

  for(size_t i=0; i<v_keys_out.size(); ++i) {
    if ( procEventForType<int16_t> (evt, i) ) { is_found = true; continue; } 
    if ( procEventForType<float>   (evt, i) ) { is_found = true; continue; } 
    if ( procEventForType<double>  (evt, i) ) { is_found = true; continue; } 
    if ( procEventForType<int>     (evt, i) ) { is_found = true; continue; } 
  }

  if(m_print_bits & 4 && !is_found && ++m_count_msg < 11) {
    MsgLog(name(), info, "ndarray is not available in the eventStore, event:" << m_count_evt 
                            << " source:" << m_source << " keys:" << m_keys_in);
    if (m_count_msg == 10) MsgLog(name(), warning, "STOP WARNINGS for source:" << m_source << " keys:" << m_keys_in);    
  }

  return is_found;
}

//--------------------

bool 
CSPad2x2NDArrReshape::procCalib(Env& env)
{  
  bool is_found = false;

  for(size_t i=0; i<v_keys_out.size(); ++i) {
    if ( procCalibForType<int16_t> (env, i) ) { is_found = true; continue; }
    if ( procCalibForType<float>   (env, i) ) { is_found = true; continue; } 
    if ( procCalibForType<double>  (env, i) ) { is_found = true; continue; } 
    if ( procCalibForType<int>     (env, i) ) { is_found = true; continue; } 
  }

  if(m_print_bits & 8 && !is_found && ++m_count_msg < 11) {
    MsgLog(name(), info, "ndarray is not available in the calibStore, event:" << m_count_evt 
                            << " source:" << m_source << " keys:" << m_keys_in);
    if (m_count_msg == 10) MsgLog(name(), warning, "STOP WARNINGS for source:" << m_source << " keys:" << m_keys_in);    
  }

  return is_found;
}

//--------------------

/*
void 
CSPad2x2NDArrReshape::checkTypeImplementation()
{  
  if ( m_outtype == "float"   ) { m_dtype = FLOAT;  return; }
  if ( m_outtype == "double"  ) { m_dtype = DOUBLE; return; } 
  if ( m_outtype == "int"     ) { m_dtype = INT;    return; } 
  if ( m_outtype == "int16"   ) { m_dtype = INT16;  return; } 
  if ( m_outtype == "int16_t" ) { m_dtype = INT16;  return; } 

  const std::string msg = "The requested data type: " + m_outtype + " is not implemented";
  MsgLog(name(), warning, msg );
  throw std::runtime_error(msg);
}
*/

//--------------------

} // namespace CSPadPixCoords
