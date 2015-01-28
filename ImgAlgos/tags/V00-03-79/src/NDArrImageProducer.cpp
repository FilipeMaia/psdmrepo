//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class NDArrImageProducer...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/NDArrImageProducer.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <time.h>
#include <sstream>   // for stringstream

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
//#include "psddl_psana/cspad2x2.ddl.h"
//#include "PSEvt/EventId.h"
#include "PSCalib/CalibFileFinder.h"
//#include "PSCalib/SegGeometryCspad2x1V1.h"

//#include "CSPadPixCoords/Image2D.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
#include <boost/lexical_cast.hpp>

// This declares this class as psana module
using namespace ImgAlgos;
using namespace std;

PSANA_MODULE_FACTORY(NDArrImageProducer)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------

NDArrImageProducer::NDArrImageProducer (const std::string& name)
  : Module(name)
  , m_calibdir()
  , m_calibgroup()
  , m_source()
  , m_inkey()
  , m_outimgkey()
  , m_outtype()
  , m_oname()
  , m_oindex()
  , m_pix_scale_size_um()
  , m_x0_off_pix()
  , m_y0_off_pix()
  , m_xy0_off_pix(0)
  , m_mode()
  , m_do_tilt()
  , m_print_bits()
  , m_count_evt(0)
  , m_count_msg(0)
{
  // get the values from configuration or use defaults
  m_calibdir         = configStr("calibdir",           "");
  m_calibgroup       = configStr("calibgroup",         "");
  m_source           = configSrc("source",             "");
  m_inkey            = configStr("key_in",             "");
  m_outimgkey        = configStr("key_out",       "image");
  m_outtype          = configStr("type_out",      "asinp");
  m_oname            = configStr("oname",              "");
  m_oindex           = config   ("oindex",              0);
  m_pix_scale_size_um= config   ("pix_scale_size_um",   0);
  m_x0_off_pix       = config   ("x0_off_pix",          0);
  m_y0_off_pix       = config   ("y0_off_pix",          0);
  m_mode             = config   ("mode",                0);
  m_do_tilt          = config   ("do_tilt",          true);
  m_print_bits       = config   ("print_bits",          0);

  //m_source = Source(m_str_src);
  stringstream ss; ss << m_source;
  m_str_src = ss.str();

  if(m_str_src.empty()) {
    const std::string msg = "NDArrImageProducer:: Input source IS NOT specified! Please define the \"source\" parameter.";
    // MsgLog(name(), error, msg);
    throw std::runtime_error(msg);
  }

  if (m_x0_off_pix or m_y0_off_pix) {
    m_xy0_off_pix = new int[2];
    m_xy0_off_pix[0] = m_x0_off_pix;
    m_xy0_off_pix[1] = m_y0_off_pix;
  }

  checkTypeImplementation();
}

//--------------
// Destructor --
//--------------

NDArrImageProducer::~NDArrImageProducer ()
{
}

//--------------------

// Print input parameters
void 
NDArrImageProducer::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n calibdir              : " << m_calibdir
        << "\n calibgroup            : " << m_calibgroup
        << "\n source                : " << m_source
        << "\n inkey                 : " << m_inkey      
        << "\n outimgkey             : " << m_outimgkey
        << "\n outtype               : " << m_outtype
        << "\n oname                 : " << m_oname            
        << "\n oindex                : " << m_oindex           
        << "\n pix_scale_size_um     : " << m_pix_scale_size_um
        << "\n x0_off_pix            : " << m_x0_off_pix       
        << "\n y0_off_pix            : " << m_y0_off_pix       
        << "\n mode                  : " << m_mode
        << "\n do_tilt               : " << m_do_tilt
        << "\n print_bits            : " << m_print_bits
        << "\n";     
  }
} 

//--------------------
/// Method which is called once at the beginning of the job
void 
NDArrImageProducer::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters(); 
  if( m_print_bits & 16) printSizeOfTypes();
}

//--------------------
/// Method which is called at the beginning of the run
void 
NDArrImageProducer::beginRun(Event& evt, Env& env)
{
  m_count_clb = 0; 
}

//--------------------
/// Method which is called at the beginning of the calibration cycle
void 
NDArrImageProducer::beginCalibCycle(Event& evt, Env& env)
{
}

//--------------------
/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
NDArrImageProducer::event(Event& evt, Env& env)
{
  ++m_count_evt; 
  if( m_print_bits & 4 ) MsgLog(name(), info, stringTimeStamp(evt) << " evt:" << m_count_evt);

  //if ( ! getCalibPars(evt, env) ) return; // skip event processing if calibration is 

  procEvent(evt, env);
}

//--------------------  
/// Method which is called at the end of the calibration cycle
void 
NDArrImageProducer::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------
/// Method which is called at the end of the run
void 
NDArrImageProducer::endRun(Event& evt, Env& env)
{
}

//--------------------
/// Method which is called once at the end of the job
void 
NDArrImageProducer::endJob(Event& evt, Env& env)
{
}

//--------------------
//--------------------
//--------------------
//--------------------

bool 
NDArrImageProducer::getCalibPars(Event& evt, Env& env)
{
  if(m_count_clb) return true;

  std::string calib_dir = (m_calibdir.empty()) ? env.calibDir() : m_calibdir;
  std::string group = (m_calibgroup.empty()) ? calibGroupForSource(m_source) : m_calibgroup; 
  unsigned prbits = (m_print_bits & 64) ? 0177777 : 0;
  int runnum = getRunNumber(evt);

  PSCalib::CalibFileFinder calibfinder(calib_dir, group, prbits);
  std::string fname = calibfinder.findCalibFile(m_src, "geometry", runnum);

  if( fname.empty() ) {
    if( m_print_bits & 2 ) MsgLog(name(), warning, "NOT FOUND \"geometry\" file for calibdir: " << calib_dir
				  << " group:" << group 
                                  << " src:"  << m_source
                                  << " run:" << runnum);
    return false;
  }
  // if "geometry" file exists - that's it!

  if( m_print_bits & 2 ) MsgLog(name(), info, "Use \"geometry\" constants for run: " << runnum 
                                              << " from file:\n" << fname);
  m_geometry = new PSCalib::GeometryAccess(fname, prbits);

  m_geometry->get_pixel_coord_indexes(m_coor_x_ind, m_coor_y_ind, m_size, 
                                      m_oname, m_oindex, m_pix_scale_size_um, m_xy0_off_pix, m_do_tilt);

  if( m_print_bits & 32 ) m_geometry->print_pixel_coords();

  m_x_ind_max = m_coor_x_ind[0];
  m_y_ind_max = m_coor_y_ind[0];
  for (unsigned i=1; i<m_size; ++i) {
    if (m_coor_x_ind[i] > m_x_ind_max) m_x_ind_max = m_coor_x_ind[i];
    if (m_coor_y_ind[i] > m_y_ind_max) m_y_ind_max = m_coor_y_ind[i];
  }

  m_count_clb++; 
  return true;
}

//--------------------
/// Do job to process event
void 
NDArrImageProducer::procEvent(Event& evt, Env& env)
{
  if ( procNDArrForType <int16_t>  (evt, env) ) return;
  if ( procNDArrForType <uint16_t> (evt, env) ) return;
  if ( procNDArrForType <float>    (evt, env) ) return;
  if ( procNDArrForType <double>   (evt, env) ) return;
  if ( procNDArrForType <int>      (evt, env) ) return;

  m_count_msg++;
  if (m_count_msg < 11 && m_print_bits) {
    MsgLog(name(), warning, "procEvent(...): Input data or geometry file is not available in event:" 
                               << m_count_evt << " for source:"
			       << m_source << " key:" << m_inkey);
    if (m_count_msg == 10) MsgLog(name(), warning, "STOP PRINT WARNINGS for source:"
			       << m_source << " key:" << m_inkey);
  }
}

//--------------------

void 
NDArrImageProducer::checkTypeImplementation()
{  
  if ( m_outtype == "asinp"  ) { m_dtype = ASINP;  return; } 
  if ( m_outtype == "int16"  ) { m_dtype = INT16;  return; } 
  if ( m_outtype == "float"  ) { m_dtype = FLOAT;  return; }
  if ( m_outtype == "double" ) { m_dtype = DOUBLE; return; } 
  if ( m_outtype == "int"    ) { m_dtype = INT;    return; } 

  const std::string msg = "The requested data type: " + m_outtype + " is not implemented";
  MsgLog(name(), warning, msg );
  throw std::runtime_error(msg);
}

//--------------------

} // namespace ImgAlgos
