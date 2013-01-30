//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CameraPixSpectra...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgPixSpectra/CameraPixSpectra.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <fstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
// to work with detector data include corresponding 
// header from psddl_psana package
// #include "psddl_psana/acqiris.ddl.h"

//#include "psddl_psana/cspad.ddl.h"
#include "PSEvt/EventId.h"
#include "CSPadPixCoords/Image2D.h"
#include "psddl_psana/opal1k.ddl.h"
#include "psddl_psana/princeton.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace ImgPixSpectra;
PSANA_MODULE_FACTORY(CameraPixSpectra)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgPixSpectra {

//----------------
// Constructors --
//----------------
CameraPixSpectra::CameraPixSpectra (const std::string& name)
  : Module(name)
  , m_src()
  , m_key()
  , m_amin()
  , m_amax()
  , m_nbins()
  , m_arr_fname()
  , m_maxEvents()
  , m_filter()
  , m_count(0)
{
  // get the values from configuration or use defaults
//m_src = configStr("source", "DetInfo(:Princeton)");
  m_src           = configStr("source", "DetInfo(SxrBeamline.0:Opal1000.1)");  
  m_key           = configStr("inputKey",   "");
  m_amin          = config   ("amin",       0.);
  m_amax          = config   ("amax",    1000.);
  m_nbins         = config   ("nbins",     100);
  m_arr_fname     = configStr("arr_fname", "camera_spectral_array.txt");
  m_maxEvents     = config   ("events", 1<<31U);
  m_filter        = config   ("filter",  false);
}

//--------------
// Destructor --
//--------------
CameraPixSpectra::~CameraPixSpectra ()
{
}

/// Method which is called once at the beginning of the job
void 
CameraPixSpectra::beginJob(Event& evt, Env& env)
{
      m_offset    = 0;
      m_width     = 0;
      m_height    = 0;
      m_numPixels = 0;

  shared_ptr<Psana::Opal1k::ConfigV1> config = env.configStore().get(m_src); 
  if (config.get()) {
      m_offset    = (int) config ->output_offset();
  }

  shared_ptr<Psana::Princeton::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1.get()) {    
      m_width     = (int) config1->width();
      m_height    = (int) config1->height();
      m_numPixels = (int) config1->numPixels();
  }

  shared_ptr<Psana::Princeton::ConfigV2> config2 = env.configStore().get(m_src);
  if (config2.get()) {    
      m_width     = (int) config2->width();
      m_height    = (int) config2->height();
      m_numPixels = (int) config2->numPixels();
  }

  WithMsgLog(name(), info, str) {
      str << "\n  offset    = " << m_offset;
      str << "\n  width     = " << m_width;
      str << "\n  height    = " << m_height;
      str << "\n  numPixels = " << m_numPixels;
  }

  this -> printInputPars();
  if( m_numPixels != 0 ) this -> arrayInit();
}

/// Method which is called at the beginning of the run
void 
CameraPixSpectra::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
CameraPixSpectra::beginCalibCycle(Event& evt, Env& env)
{

  shared_ptr<Psana::Opal1k::ConfigV1> config = env.configStore().get(m_src); 
  if (config.get()) {
    WithMsgLog(name(), info, str) {
      str << "Psana::Opal1k::ConfigV1:";
      str << "\n  black_level = "                     << config->black_level();
      str << "\n  gain_percent = "                    << config->gain_percent();
      str << "\n  output_resolution = "               << config->output_resolution();
      str << "\n  vertical_binning = "                << config->vertical_binning();
      str << "\n  output_mirroring = "                << config->output_mirroring();
      str << "\n  vertical_remapping = "              << int(config->vertical_remapping());
      str << "\n  output_offset = "                   << config->output_offset();
      str << "\n  output_resolution_bits = "          << config->output_resolution_bits();
      str << "\n  defect_pixel_correction_enabled = " << int(config->defect_pixel_correction_enabled());
      str << "\n  output_lookup_table_enabled = "     << int(config->output_lookup_table_enabled());
    }
  }

  shared_ptr<Psana::Princeton::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1.get()) {    
    WithMsgLog(name(), info, str) {
      str << "Princeton::ConfigV1:";
      str << "\n  width = "              << config1->width();
      str << "\n  height = "             << config1->height();
      str << "\n  orgX = "               << config1->orgX();
      str << "\n  orgY = "               << config1->orgY();
      str << "\n  binX = "               << config1->binX();
      str << "\n  binY = "               << config1->binY();
      str << "\n  exposureTime = "       << config1->exposureTime();
      str << "\n  coolingTemp = "        << config1->coolingTemp();
      str << "\n  readoutSpeedIndex = "  << config1->readoutSpeedIndex();
      str << "\n  readoutEventCode = "   << config1->readoutEventCode();
      str << "\n  delayMode = "          << config1->delayMode();
      str << "\n  frameSize = "          << config1->frameSize();
      str << "\n  numPixels = "          << config1->numPixels();

    }
  }
 
  shared_ptr<Psana::Princeton::ConfigV2> config2 = env.configStore().get(m_src);
  if (config2.get()) {    
    WithMsgLog(name(), info, str) {
      str << "Princeton::ConfigV2:";
      str << "\n  width = "              << config2->width();
      str << "\n  height = "             << config2->height();
      str << "\n  orgX = "               << config2->orgX();
      str << "\n  orgY = "               << config2->orgY();
      str << "\n  binX = "               << config2->binX();
      str << "\n  binY = "               << config2->binY();
      str << "\n  exposureTime = "       << config2->exposureTime();
      str << "\n  coolingTemp = "        << config2->coolingTemp();
      str << "\n  gainIndex = "          << config2->gainIndex();
      str << "\n  readoutSpeedIndex = "  << config2->readoutSpeedIndex();
      str << "\n  readoutEventCode = "   << config2->readoutEventCode();
      str << "\n  delayMode = "          << config2->delayMode();
      str << "\n  frameSize = "          << config2->frameSize();
      str << "\n  numPixels = "          << config2->numPixels();
    }    
  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CameraPixSpectra::event(Event& evt, Env& env)
{
  //// example of getting non-detector data from event
  //shared_ptr<PSEvt::EventId> eventId = evt.get();
  //if (eventId.get()) {
  //  // example of producing messages using MgsLog facility
  //  MsgLog(name(), info, "event ID: " << *eventId);
  //}
  
  //// tis is how to skip event (all downstream modules will not be called)
  //if (m_filter && m_count % 10 == 0) { skip(); return; }
  
  // this is how to gracefully stop analysis job
  if (m_count >= m_maxEvents) { stop(); return; }

  if (   m_count<5 
     or (m_count<500 and m_count%100  == 0) 
     or                  m_count%1000 == 0  ) WithMsgLog(name(), info, log) { log << "event=" << m_count; }


  // Camera::FrameV1 (for Camera, Opal1k, etc.)
  shared_ptr<Psana::Camera::FrameV1> frmData = evt.get(m_src);
  if (frmData.get()) {

      if (m_numPixels == 0) {
        m_offset    = (int) frmData->offset();
        m_width     = (int) frmData->width();
        m_height    = (int) frmData->height();
        m_numPixels = m_width * m_height;
        this -> arrayInit();
      }

      const ndarray<const uint8_t,2>& data8 = frmData->data8();
      if (not data8.empty()) {
          const uint8_t *p_data = &data8[0][0];
          this -> arrayFill8 (p_data);
      }

      const ndarray<const uint16_t,2>& data16 = frmData->data16();
      if (not data16.empty()){
          const uint16_t *p_data = &data16[0][0];
          this -> arrayFill (p_data);
      }

    /*
    WithMsgLog(name(), info, str) {
      str << "Camera::FrameV1: width=" << frmData->width()
          << " height="                << frmData->height()
          << " depth="                 << frmData->depth()
          << " offset="                << frmData->offset() ;

      const ndarray<const uint8_t,2>& data8 = frmData->data8();
      if (not data8.empty()) {
        str << " data8=["  << int(data8[0][0])
            << ", "        << int(data8[0][1])
            << ", "        << int(data8[0][2]) << ", ...]";
      }

      const ndarray<const uint16_t,2>& data16 = frmData->data16();
      if (not data16.empty()) {
        str << " data16=[" << int(data16[0][0])
            << ", "        << int(data16[0][1])
            << ", "        << int(data16[0][2]) << ", ...]";
      }
    }
    */
  }

  // Princeton::FrameV1
  shared_ptr<Psana::Princeton::FrameV1> frame = evt.get(m_src);
  if (frame.get()) {

    const ndarray<const uint16_t,2>& data = frame->data();
    const uint16_t *p_data = &data[0][0];
    this -> arrayFill (p_data);

    /*
    WithMsgLog(name(), info, str) {
      str << "Princeton::FrameV1:";
      str << "\n  shotIdStart = " << frame->shotIdStart();
      str << "\n  readoutTime = " << frame->readoutTime();

    const ndarray<const uint16_t,2>& data = frame->data();
      str << "\n  data =";
      for (int i = 0; i < 10; ++ i) {
        str << " " << data[0][i];
      }
      str << " ...";
    }
    */
  }

  // increment event counter
  ++ m_count;
}

  
/// Method which is called at the end of the calibration cycle
void 
CameraPixSpectra::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
CameraPixSpectra::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
CameraPixSpectra::endJob(Event& evt, Env& env)
{
  MsgLog(name(), info, "CameraPixSpectra::endJob");
  this -> saveArrayInFile();
  this -> saveShapeInFile();
  this -> arrayDelete();
}

//--------------------
//--------------------
//--------------------
//--------------------

void 
CameraPixSpectra::arrayInit()
{
  MsgLog(name(), info, "CameraPixSpectra::arrayInit()");

  m_factor = double(m_nbins) / (m_amax-m_amin);  // scale factor for histogramm index
  m_nbins1 = m_nbins - 1;

  int size = m_numPixels * m_nbins;
  m_arr    = new int [size];
  for(int i=0; i<size; i++) m_arr[i] = 0;
}

//--------------------

void 
CameraPixSpectra::arrayDelete()
{
  delete [] m_arr;
}

//--------------------

void
CameraPixSpectra::arrayFill(const uint16_t* data)
{
             for (int i=0; i<m_numPixels; i++) {
               double amp = (double)data[i];
               int iamp = this -> ampToIndex(amp);
	       //cout << "pix=" << i << " amp=" << amp << endl;  
               m_arr[i * m_nbins + iamp] ++; // incriment in spectral array
             }
}

//--------------------

void
CameraPixSpectra::arrayFill8(const uint8_t* data)
{
             for (int i=0; i<m_numPixels; i++) {
               double amp = (double)data[i];
               int iamp = this -> ampToIndex(amp);
	       //cout << "pix=" << i << " amp=" << amp << endl;  
               m_arr[i * m_nbins + iamp] ++; // incriment in spectral array
             }
}

//--------------------

void 
CameraPixSpectra::saveArrayInFile()
{ 
    MsgLog(name(), info, "Save the spectral array in file " << m_arr_fname);
    CSPadPixCoords::Image2D<int>* arr = new CSPadPixCoords::Image2D<int>(&m_arr[0], m_numPixels, m_nbins); 
    arr -> saveImageInFile(m_arr_fname,0);
}

//--------------------

void 
CameraPixSpectra::saveShapeInFile()
{ 
    m_arr_shape_fname = m_arr_fname + ".sha";
    MsgLog(name(), info, "Save the spectral array configuration in file " << m_arr_shape_fname);
    ofstream file; 
    file.open(m_arr_shape_fname.c_str(), std::ios_base::out);
    file << "NPIXELS  " << m_numPixels       << "\n";
    file << "NBINS    " << m_nbins           << "\n";
    file << "AMIN     " << m_amin            << "\n";
    file << "AMAX     " << m_amax            << "\n";
    file << "NEVENTS  " << m_count           << "\n";
    file << "ARRFNAME " << m_arr_fname       << "\n";
    file.close();
}

//--------------------

int  
CameraPixSpectra::ampToIndex(double amp)
{
    int ind = (int) (m_factor*(amp-m_amin));
    if( ind < 0       ) return 0;
    if( ind > m_nbins1) return m_nbins1;
    return ind;
}

//--------------------

void 
CameraPixSpectra::printInputPars()
{
  WithMsgLog(name(), info, log) { log 
        << "\n    Input parameters:"
      //<< "\n    m_src         " << m_src       
        << "\n    m_key         " << m_key    
        << "\n    m_maxEvents   " << m_maxEvents 
        << "\n    m_amin        " << m_amin      
        << "\n    m_amax        " << m_amax      
        << "\n    m_nbins       " << m_nbins     
        << "\n    m_arr_fname   " << m_arr_fname    
      //<< "\n    m_filter      " << m_filter
        << "\n";
      }
}

//--------------------

} // namespace ImgPixSpectra

