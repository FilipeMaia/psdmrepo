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
  , m_tiltIsApplied()
  , m_useWidePixCenter()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_calibDir         = configStr("calibDir",      ""); // if not provided, default from env will be used
  m_typeGroupName    = configStr("typeGroupName", "CsPad2x2::CalibV1");
  m_source           = configSrc("source",        "DetInfo(:Cspad2x2.1)");
  m_inkey            = configStr("inkey",         "");
  m_outimgkey        = configStr("outimgkey",     "image");
  m_tiltIsApplied    = config   ("tiltIsApplied",    true);
  m_useWidePixCenter = config   ("useWidePixCenter",false);
  m_print_bits       = config   ("print_bits",          0);
  //m_source = Source(m_str_src);
  //stringstream ssrc; ssrc << m_source;
  //m_str_src = ssrc.str();
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
}

//--------------------
/// Method which is called at the beginning of the run
void 
CSPad2x2ImageProducer::beginRun(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) MsgLog(name(), info, "In beginRun(...)");

  this -> getConfigPars(env);      // get m_src here
  this -> getCalibPars(evt, env);  // use m_src here
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
  ++m_count; //cout << "Event: " << m_count;
  if( m_print_bits & 2 ) printTimeStamp(evt);

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

/// Return the run number
int 
CSPad2x2ImageProducer::getRunNumber(Event& evt)
{
  shared_ptr<EventId> eventId = evt.get();
  if (eventId.get()) {
    return eventId->run();
  } else {
    MsgLog(name(), warning, "Cannot determine run number, will use 0.");
    return int(0);
  }
}

//--------------------

void 
CSPad2x2ImageProducer::getConfigPars(Env& env)
{
  m_count_cfg = 0; 
  if ( getConfigParsForType <Psana::CsPad2x2::ConfigV1> (env) ) return;
  if ( getConfigParsForType <Psana::CsPad2x2::ConfigV2> (env) ) return;

  MsgLog(name(), error, "No CsPad2x2 configuration objects found, terminating.");
  terminate();
}

//--------------------

void 
CSPad2x2ImageProducer::getCalibPars(Event& evt, Env& env)
{
  std::string calib_dir = (m_calibDir == "") ? env.calibDir() : m_calibDir;
  m_cspad2x2_calibpars = new PSCalib::CSPad2x2CalibPars(calib_dir, m_typeGroupName, m_src, getRunNumber(evt));

  m_pix_coords_cspad2x2 = new PC2X2 (m_cspad2x2_calibpars, m_tiltIsApplied, m_useWidePixCenter);

  if( m_print_bits & 1 ) {
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
  shared_ptr<Psana::CsPad2x2::ElementV1> elem1 = evt.get(m_source, m_inkey, &m_src); // get m_src here

  if (elem1) {

    for (unsigned i=0; i<PC2X2::N2X1_IN_DET; i++) m_common_mode[i] = elem1->common_mode(i);

    const ndarray<const int16_t, 3>& data_nda = elem1->data();
    //const int16_t* data = &data_nda[0][0][0];

    this -> cspad_image_fill (data_nda);
    this -> cspad_image_add_in_event(evt);
  } // if (elem1)
}

//--------------------

void
CSPad2x2ImageProducer::cspad_image_fill(const ndarray<const int16_t,3>& data)
{
  std::fill_n(&m_arr_cspad2x2_image[0][0], int(NX_CSPAD2X2*NY_CSPAD2X2), double(0));

  for(unsigned sect=0; sect < PC2X2::N2X1_IN_DET; ++sect) {
    if ( !(m_roiMask & (1<<sect)) ) continue;
 
      for (unsigned r=0; r<PC2X2::ROWS2X1; ++r) {
      for (unsigned c=0; c<PC2X2::COLS2X1; ++c) {

        int ix = int (m_pix_coords_cspad2x2 -> getPixCoor_um (PC2X2::AXIS_X, sect, r, c) * PC2X2::UM_TO_PIX);
        int iy = int (m_pix_coords_cspad2x2 -> getPixCoor_um (PC2X2::AXIS_Y, sect, r, c) * PC2X2::UM_TO_PIX);

        if(ix <  0)           continue;
        if(iy <  0)           continue;
        if(ix >= NX_CSPAD2X2) continue;
        if(iy >= NY_CSPAD2X2) continue;

        m_arr_cspad2x2_image[ix][iy] += (double)data[r][c][sect]; 
      }
      }
  }
}

//--------------------

void
CSPad2x2ImageProducer::cspad_image_save_in_file(const std::string &filename)
{
  CSPadPixCoords::Image2D<double> *img2d = new CSPadPixCoords::Image2D<double>(&m_arr_cspad2x2_image[0][0], NY_CSPAD2X2, NX_CSPAD2X2);
  img2d -> saveImageInFile(filename,0);
}

//--------------------
void
CSPad2x2ImageProducer::cspad_image_add_in_event(Event& evt)
{
  if(m_outimgkey == "Image2D") {

    shared_ptr< CSPadPixCoords::Image2D<double> > img2d( new CSPadPixCoords::Image2D<double>(&m_arr_cspad2x2_image[0][0], NY_CSPAD2X2, NX_CSPAD2X2) );
    evt.put(img2d, m_src, m_outimgkey);

  } else {

    const unsigned shape[] = {NY_CSPAD2X2, NX_CSPAD2X2};
    shared_ptr< ndarray<double,2> > img2d( new ndarray<double,2>(&m_arr_cspad2x2_image[0][0],shape) );
    evt.put(img2d, m_src, m_outimgkey);
  }
}

//--------------------

void 
CSPad2x2ImageProducer::printTimeStamp(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {

    MsgLog( name(), info, " Run="   <<  eventId->run()
                       << " Event=" <<  m_count 
                       << " Time="  <<  eventId->time() );
  }
}

//--------------------

} // namespace CSPadPixCoords
