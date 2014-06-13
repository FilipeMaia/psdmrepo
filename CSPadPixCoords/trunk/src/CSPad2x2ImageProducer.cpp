//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPad2x2ImageProducer...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadPixCoords/CSPad2x2ImageProducer.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <time.h>
#include <sstream>   // for stringstream

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psddl_psana/cspad2x2.ddl.h"
#include "PSEvt/EventId.h"

#include "CSPadPixCoords/Image2D.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
#include <boost/lexical_cast.hpp>

// This declares this class as psana module
using namespace CSPadPixCoords;
using namespace std;

PSANA_MODULE_FACTORY(CSPad2x2ImageProducer)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace CSPadPixCoords {

//----------------
// Constructors --
//----------------

CSPad2x2ImageProducer::CSPad2x2ImageProducer (const std::string& name)
  : Module(name)
  , m_calibDir()
  , m_typeGroupName()
  , m_source()
  , m_inkey()
  , m_outimgkey()
  , m_outtype()
  , m_tiltIsApplied()
  , m_useWidePixCenter()
  , m_print_bits()
  , m_count(0)
  , m_count_msg(0)
{
  // get the values from configuration or use defaults
  m_calibDir         = configStr("calibDir",      ""); // if not provided, default from env will be used
  m_typeGroupName    = configStr("typeGroupName", "CsPad2x2::CalibV1");
  m_source           = configSrc("source",        "DetInfo(:Cspad2x2.1)");
  m_inkey            = configStr("inkey",         "");
  m_outimgkey        = configStr("outimgkey",     "image");
  m_outtype          = configStr("outtype",       "int16");
  m_tiltIsApplied    = config   ("tiltIsApplied",    true);
  m_useWidePixCenter = config   ("useWidePixCenter",false);
  m_print_bits       = config   ("print_bits",          0);
  //m_source = Source(m_str_src);
  //stringstream ssrc; ssrc << m_source;
  //m_str_src = ssrc.str();

  checkTypeImplementation();
}

//--------------
// Destructor --
//--------------

CSPad2x2ImageProducer::~CSPad2x2ImageProducer ()
{
}

//--------------------

// Print input parameters
void 
CSPad2x2ImageProducer::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n calibDir              : " << m_calibDir
        << "\n typeGroupName         : " << m_typeGroupName
        << "\n source                : " << m_source
        << "\n inkey                 : " << m_inkey      
        << "\n outimgkey             : " << m_outimgkey
        << "\n outtype               : " << m_outtype
        << "\n tiltIsApplied         : " << m_tiltIsApplied
        << "\n useWidePixCenter      : " << m_useWidePixCenter
        << "\n print_bits            : " << m_print_bits
        << "\n";     
  }

  MsgLog(name(), debug, 
           "\n NX_CSPAD2X2 : "      << NX_CSPAD2X2  
        << "\n NY_CSPAD2X2 : "      << NY_CSPAD2X2
        << "\n"
        );
} 

//--------------------
/// Method which is called once at the beginning of the job
void 
CSPad2x2ImageProducer::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters(); 
  if( m_print_bits & 16) printSizeOfTypes();
}

//--------------------
/// Method which is called at the beginning of the run
void 
CSPad2x2ImageProducer::beginRun(Event& evt, Env& env)
{
  m_count_cfg = 0; 
}

//--------------------
/// Method which is called at the beginning of the calibration cycle
void 
CSPad2x2ImageProducer::beginCalibCycle(Event& evt, Env& env)
{
}

//--------------------
/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPad2x2ImageProducer::event(Event& evt, Env& env)
{
  ++m_count;
  if( m_print_bits & 4 ) printTimeStamp(evt, m_count);

  if ( m_count_cfg==0 ) {
    getConfigPars(env);           // get m_src here
    if ( m_count_cfg==0 ) return; // skip event processing if configuration is missing
    getCalibPars(evt, env);       // use m_src here
  }

  processEvent(evt, env);
}

//--------------------  
/// Method which is called at the end of the calibration cycle
void 
CSPad2x2ImageProducer::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------
/// Method which is called at the end of the run
void 
CSPad2x2ImageProducer::endRun(Event& evt, Env& env)
{
}

//--------------------
/// Method which is called once at the end of the job
void 
CSPad2x2ImageProducer::endJob(Event& evt, Env& env)
{
}

//--------------------
//--------------------
//--------------------
//--------------------

void 
CSPad2x2ImageProducer::getConfigPars(Env& env)
{
  if ( getConfigParsForType <Psana::CsPad2x2::ConfigV1> (env) ) return;
  if ( getConfigParsForType <Psana::CsPad2x2::ConfigV2> (env) ) return;

  m_count_msg++;
  if (m_count_msg < 20) MsgLog(name(), warning, "No CsPad2x2 configuration objects found. event:"<< m_count << " for source:" << m_source);
  if (m_count_msg ==20) MsgLog(name(), warning, "STOP PRINTING WARNINGS for source:" << m_source);
  //terminate();
}

//--------------------

void 
CSPad2x2ImageProducer::getCalibPars(Event& evt, Env& env)
{
  std::string calib_dir = (m_calibDir == "") ? env.calibDir() : m_calibDir;
  m_cspad2x2_calibpars = new PSCalib::CSPad2x2CalibPars(calib_dir, m_typeGroupName, m_src, getRunNumber(evt));

  m_pix_coords_cspad2x2 = new PC2X2 (m_cspad2x2_calibpars, m_tiltIsApplied, m_useWidePixCenter);

  if( m_print_bits & 2 ) {
    m_cspad2x2_calibpars  -> printInputPars();
    m_cspad2x2_calibpars  -> printCalibPars();
    //m_pix_coords_cspad2x2 -> printCoordArray(); 
    //m_pix_coords_cspad2x2 -> printConstants(); 
  }
}

//--------------------
/// Do job to process event
void 
CSPad2x2ImageProducer::processEvent(Event& evt, Env& env)
{
  // Check if the requested src and key are consistent with Psana::CsPad2x2::ElementV1
  if ( procCSPad2x2DataForType <Psana::CsPad2x2::ElementV1> (evt) ) return;

  // Check if the requested src and key are consistent with ndarray<T,3> of shape [N][185][388]
  if ( procCSPad2x2NDArrForType <float>    (evt) ) return;
  if ( procCSPad2x2NDArrForType <double>   (evt) ) return;
  if ( procCSPad2x2NDArrForType <int>      (evt) ) return;
  if ( procCSPad2x2NDArrForType <int16_t>  (evt) ) return;
  if ( procCSPad2x2NDArrForType <uint16_t> (evt) ) return;

  m_count_msg++;
  if (m_count_msg < 20) MsgLog(name(), warning, "processEvent(...): cspad2x2 data or ndarr is not available in event:" << m_count << " for source:"
			       << m_source << " key:" << m_inkey);
  if (m_count_msg ==20) MsgLog(name(), warning, "STOP PRINTING WARNINGS for source:"
			       << m_source << " key:" << m_inkey);
}

//--------------------

void
CSPad2x2ImageProducer::cspad_image_add_in_event(Event& evt)
{
  // Save image in the event for one of the supported data types
  if      ( m_outtype == "float"   ) save2DArrayInEventForType<float>   (evt); 
  else if ( m_outtype == "double"  ) save2DArrayInEventForType<double>  (evt); 
  else if ( m_outtype == "int"     ) save2DArrayInEventForType<int>     (evt); 
  else if ( m_outtype == "int16"   ) save2DArrayInEventForType<int16_t> (evt); 
  else if ( m_outtype == "int16_t" ) save2DArrayInEventForType<int16_t> (evt); 
}

//--------------------

void 
CSPad2x2ImageProducer::checkTypeImplementation()
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

//--------------------

} // namespace CSPadPixCoords
