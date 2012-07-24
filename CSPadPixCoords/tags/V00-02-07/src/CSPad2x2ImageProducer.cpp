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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
// to work with detector data include corresponding 
// header from psddl_psana package
// #include "psddl_psana/acqiris.ddl.h"
#include "psddl_psana/cspad2x2.ddl.h"
#include "PSEvt/EventId.h"

#include "CSPadPixCoords/Image2D.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
#include <boost/lexical_cast.hpp>

// This declares this class as psana module
using namespace CSPadPixCoords;
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
//, m_calibDir()
//, m_typeGroupName()
//, m_source()
  , m_str_src()
  , m_inkey()
  , m_outimgkey()
  , m_tiltIsApplied()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  //m_calibDir      = configStr("calibDir",      ""); // if not provided default from env will be used
  //m_typeGroupName = configStr("typeGroupName", "CsPad::CalibV1");

  m_str_src       = configStr("source",        "DetInfo(:Cspad2x2)");
  m_inkey         = configStr("inkey",         "");
  m_outimgkey     = configStr("outimgkey",     "Image2D");
  m_tiltIsApplied = config   ("tiltIsApplied", true);
  m_print_bits    = config   ("print_bits",    0);
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
        << "\n str_src               : " << m_str_src
        << "\n inkey                 : " << m_inkey      
        << "\n outimgkey             : " << m_outimgkey
        << "\n tiltIsApplied         : " << m_tiltIsApplied
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
  if( m_print_bits & 1<<1 ) MsgLog(name(), info, "ImageCSPad::beginRun ");


  //std::string calib_dir = (m_calibDir == "") ? env.calibDir() : m_calibDir;

  //m_cspad_calibpar   = new PSCalib::CSPadCalibPars(calib_dir, m_typeGroupName, m_str_src, getRunNumber(evt));
  m_pix_coords_2x1   = new CSPadPixCoords::PixCoords2x1   ();
  //m_pix_coords_quad  = new CSPadPixCoords::PixCoordsQuad  ( m_pix_coords_2x1,  m_cspad_calibpar, m_tiltIsApplied );
  //m_pix_coords_cspad = new CSPadPixCoords::PixCoordsCSPad ( m_pix_coords_quad, m_cspad_calibpar, m_tiltIsApplied );
  m_pix_coords_cspad2x2 = new CSPadPixCoords::PixCoordsCSPad2x2 (m_pix_coords_2x1, m_tiltIsApplied);

  //if( m_print_bits & 1<<0 ) m_cspad_calibpar  -> printCalibPars();
  m_pix_coords_2x1  -> print_member_data();
  //m_pix_coords_quad -> print_member_data(); 

  this -> getConfigPars(env);
}

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
  shared_ptr<Psana::CsPad2x2::ConfigV1> config = env.configStore().get(m_str_src);
  if (config.get()) {
    m_roiMask        = config->roiMask();
    m_numAsicsStored = config->numAsicsStored();
    WithMsgLog(name(), info, str) {
      str << "CsPad2x2::ConfigV1:";
      str << "\n  concentratorVersion = " << config->concentratorVersion();
      str << "\n  roiMask = "             << config->roiMask();
      str << "\n  numAsicsStored = "      << config->numAsicsStored();
     }  
  }

  m_n2x1         = 2;                                // 2
  m_ncols2x1     = Psana::CsPad::ColumnsPerASIC;     // 185
  m_nrows2x1     = Psana::CsPad::MaxRowsPerASIC * 2; // 388
  m_sizeOf2x1Img = m_nrows2x1 * m_ncols2x1;          // 185*388;

  XCOOR = CSPadPixCoords::PixCoords2x1::X;
  YCOOR = CSPadPixCoords::PixCoords2x1::Y;
  ZCOOR = CSPadPixCoords::PixCoords2x1::Z;
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
/// Do job to process event
void 
CSPad2x2ImageProducer::processEvent(Event& evt, Env& env)
{
  shared_ptr<Psana::CsPad2x2::ElementV1> elem1 = evt.get(m_str_src, m_inkey, &m_actualSrc); // get m_actualSrc here

  if (elem1.get()) {

    for (unsigned i=0; i<m_n2x1; i++) { m_common_mode[i] = elem1->common_mode(i); }

    const ndarray<int16_t, 3>& data_nda = elem1->data();
    //const int16_t* data = &data_nda[0][0][0];

    this -> cspad_image_fill (data_nda);
    this -> cspad_image_add_in_event(evt);
  } // if (elem1.get())
}

//--------------------
void
CSPad2x2ImageProducer::cspad_image_fill(const ndarray<int16_t,3>& data)
{
  std::fill_n(&m_arr_cspad2x2_image[0][0], int(NX_CSPAD2X2*NY_CSPAD2X2), double(0));

  for(uint32_t sect=0; sect < m_n2x1; sect++)
  {
    if (m_roiMask & (1<<sect)) {
 
      for (uint32_t c=0; c<m_ncols2x1; c++) {
      for (uint32_t r=0; r<m_nrows2x1; r++) {

        int ix = (int) m_pix_coords_cspad2x2 -> getPixCoor_pix (XCOOR, sect, r, c);
        int iy = (int) m_pix_coords_cspad2x2 -> getPixCoor_pix (YCOOR, sect, r, c);

        if(ix <  0)           continue;
        if(iy <  0)           continue;
        if(ix >= NX_CSPAD2X2) continue;
        if(iy >= NY_CSPAD2X2) continue;

        m_arr_cspad2x2_image[ix][iy] += (double)data[c][r][sect]; 
      }
      }
    }
  }
}

//--------------------
void
CSPad2x2ImageProducer::cspad_image_save_in_file(const std::string &filename)
{
  CSPadPixCoords::Image2D<double> *img2d = new CSPadPixCoords::Image2D<double>(&m_arr_cspad2x2_image[0][0],NY_CSPAD2X2,NX_CSPAD2X2);
  img2d -> saveImageInFile(filename,0);
}

//--------------------
void
CSPad2x2ImageProducer::cspad_image_add_in_event(Event& evt)
{
  if(m_outimgkey == "Image2D") {

    shared_ptr< CSPadPixCoords::Image2D<double> > img2d( new CSPadPixCoords::Image2D<double>(&m_arr_cspad2x2_image[0][0],NY_CSPAD2X2,NX_CSPAD2X2) );
    evt.put(img2d, m_actualSrc, m_outimgkey);

  } else {

    const unsigned shape[] = {NY_CSPAD2X2,NX_CSPAD2X2};
    shared_ptr< ndarray<double,2> > img2d( new ndarray<double,2>(&m_arr_cspad2x2_image[0][0],shape) );
    evt.put(img2d, m_actualSrc, m_outimgkey);
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
