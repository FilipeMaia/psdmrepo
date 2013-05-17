//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadImageProducer...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadPixCoords/CSPadImageProducer.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <time.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
#include <boost/lexical_cast.hpp>

// This declares this class as psana module
using namespace CSPadPixCoords;
PSANA_MODULE_FACTORY(CSPadImageProducer)

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace CSPadPixCoords {

//----------------
// Constructors --
//----------------

CSPadImageProducer::CSPadImageProducer (const std::string& name)
  : Module(name)
  , m_calibDir()
  , m_typeGroupName()
  , m_str_src()
  , m_inkey()
  , m_imgkey()
  , m_tiltIsApplied()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_calibDir      = configStr("calibDir",      ""); // if not provided default from env will be used
  m_typeGroupName = configStr("typeGroupName", "CsPad::CalibV1");
  m_str_src       = configStr("source",        "CxiDs1.0:Cspad.0");
  m_inkey         = configStr("key",           "");
  m_imgkey        = configStr("imgkey",        "Image2D");
  m_tiltIsApplied = config   ("tiltIsApplied", true);
  m_print_bits    = config   ("print_bits",    0);

  m_source        = Source(m_str_src);
}


//--------------
// Destructor --
//--------------

CSPadImageProducer::~CSPadImageProducer ()
{
}

//--------------------

/// Print input parameters
void 
CSPadImageProducer::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\nInput parameters:"
        << "\ncalibDir      : "     << m_calibDir     
        << "\ntypeGroupName : "     << m_typeGroupName
        << "\nstr_src       : "     << m_str_src      
        << "\nsource        : "     << m_source      
        << "\nkey           : "     << m_inkey        
        << "\nimgkey        : "     << m_imgkey       
        << "\ntiltIsApplied : "     << m_tiltIsApplied
        << "\nprint_bits    : "     << m_print_bits;
  }
}

//--------------------

/// Method which is called once at the beginning of the job
void 
CSPadImageProducer::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
}

//--------------------

/// Method which is called at the beginning of the run
void 
CSPadImageProducer::beginRun(Event& evt, Env& env)
{
  // get run number
  shared_ptr<EventId> eventId = evt.get();
  int run = 0;
  if (eventId.get()) {
    run = eventId->run();
  } else {
    MsgLog(name(), warning, "Cannot determine run number, will use 0.");
  }

  std::string calib_dir = (m_calibDir == "") ? env.calibDir() : m_calibDir;

  //m_cspad_calibpar = new PSCalib::CSPadCalibPars(); // get default calib pars from my local directory
                                                      // ~dubrovin/LCLS/CSPadAlignment-v01/calib-cxi35711-r0009-det/
  m_cspad_calibpar   = new PSCalib::CSPadCalibPars(calib_dir, m_typeGroupName, m_str_src, run);
  m_pix_coords_2x1   = new CSPadPixCoords::PixCoords2x1   ();
  m_pix_coords_quad  = new CSPadPixCoords::PixCoordsQuad  ( m_pix_coords_2x1,  m_cspad_calibpar, m_tiltIsApplied );
  m_pix_coords_cspad = new CSPadPixCoords::PixCoordsCSPad ( m_pix_coords_quad, m_cspad_calibpar, m_tiltIsApplied );

  m_coor_x_pix = m_pix_coords_cspad -> getPixCoorArrX_pix();
  m_coor_y_pix = m_pix_coords_cspad -> getPixCoorArrY_pix();
  m_coor_x_int = m_pix_coords_cspad -> getPixCoorArrX_int();
  m_coor_y_int = m_pix_coords_cspad -> getPixCoorArrY_int();

  if( m_print_bits & 2 ) m_cspad_calibpar  -> printCalibPars();
  //m_pix_coords_2x1  -> print_member_data();
  //m_pix_coords_quad -> print_member_data(); 

  this -> getQuadConfigPars(env);
}

//--------------------

void 
CSPadImageProducer::getQuadConfigPars(Env& env)
{
  m_n2x1         = Psana::CsPad::SectorsPerQuad;     // 8
  m_ncols2x1     = Psana::CsPad::ColumnsPerASIC;     // 185
  m_nrows2x1     = Psana::CsPad::MaxRowsPerASIC * 2; // 388
  m_sizeOf2x1Img = m_nrows2x1 * m_ncols2x1;          // 185*388;

  XCOOR = CSPadPixCoords::PixCoords2x1::X;
  YCOOR = CSPadPixCoords::PixCoords2x1::Y;
  ZCOOR = CSPadPixCoords::PixCoords2x1::Z;

  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV2>(env) ) return;
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV3>(env) ) return;
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV4>(env) ) return;
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV5>(env) ) return;

  MsgLog(name(), warning, "CsPad::ConfigV2 - V5 is not available in this run.");
}

//--------------------

/// Method which is called at the beginning of the calibration cycle
void 
CSPadImageProducer::beginCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPadImageProducer::event(Event& evt, Env& env)
{
  // this is how to gracefully stop analysis job
  ++m_count;

  struct timespec start, stop;
  int status = clock_gettime( CLOCK_REALTIME, &start ); // Get LOCAL time

  procEvent(evt, env);

  if( m_print_bits & 4 ) {
    status = clock_gettime( CLOCK_REALTIME, &stop ); // Get LOCAL time
    cout << "  Time to produce cspad image is " 
         << stop.tv_sec - start.tv_sec + 1e-9*(stop.tv_nsec - start.tv_nsec) 
         << " sec" << endl;
  }
}

//--------------------
  
/// Method which is called at the end of the calibration cycle
void 
CSPadImageProducer::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the end of the run
void 
CSPadImageProducer::endRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called once at the end of the job
void 
CSPadImageProducer::endJob(Event& evt, Env& env)
{
}

//--------------------

void 
CSPadImageProducer::getCSPadConfigFromData(Event& evt)
{
  if ( getCSPadConfigFromDataForType <Psana::CsPad::DataV1, Psana::CsPad::ElementV1> (evt) ) return;
  if ( getCSPadConfigFromDataForType <Psana::CsPad::DataV2, Psana::CsPad::ElementV2> (evt) ) return;

  MsgLog(name(), warning, "Psana::CsPad::DataV# / ElementV# for #=[2-5] is not available in this run.");
}

//--------------------

void 
CSPadImageProducer::procEvent(Event& evt, Env& env)
{  
  getCSPadConfigFromData(evt);

  // Check if the requested src and key are consistent with Psana::CsPad::DataV1, or V2
  if ( procCSPadDataForType  <Psana::CsPad::DataV1, Psana::CsPad::ElementV1> (evt) ) return;
  if ( procCSPadDataForType  <Psana::CsPad::DataV2, Psana::CsPad::ElementV2> (evt) ) return;

  // Check if the requested src and key are consistent with ndarray<T,3> of shape [N][185][388]
  if ( procCSPadNDArrForType <float>    (evt) ) return;
  //if ( procCSPadNDArrForType <double>   (evt) ) return;
  //if ( procCSPadNDArrForType <uint16_t> (evt) ) return;
  //if ( procCSPadNDArrForType <int>      (evt) ) return;

  MsgLog(name(), warning, "Psana::CsPad::DataV# / ElementV# for #=[2-5] is not available in this run.");
}

//--------------------

} // namespace CSPadPixCoords
