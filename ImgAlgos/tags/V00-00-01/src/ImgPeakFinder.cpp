//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgPeakFinder...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/ImgPeakFinder.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <math.h> // for exp

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
//#include "psddl_psana/acqiris.ddl.h"
#include "psddl_psana/camera.ddl.h"
#include "PSEvt/EventId.h"
//#include "PSTime/Time.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
#include <iomanip> // for setw, setfill
#include <sstream> // for streamstring


// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(ImgPeakFinder)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
ImgPeakFinder::ImgPeakFinder (const std::string& name)
  : Module(name)
  , m_src()
  , m_key()
  , m_peaksKey()
  , m_thr_low()
  , m_thr_high()
  , m_sigma()
  , m_nsm()
  , m_npeak()
  , m_xmin()
  , m_xmax()
  , m_ymin()
  , m_ymax()
  , m_filter()
  , m_event()
  , m_count(0)
  , m_selected(0)
{
  // get the values from configuration or use defaults
  m_src        = configStr("source",  "DetInfo()");
  m_key        = configStr("key",              "");
  m_peaksKey   = configStr("peaksKey",    "peaks");
  m_thr_low    = config   ("threshold_low",    10);
  m_thr_high   = config   ("threshold_high",  100);
  m_sigma      = config   ("sigma",           1.5);
  m_nsm        = config   ("smear_radius",      1);
  m_npeak      = config   ("peak_radius",       3);
  m_xmin       = config   ("xmin",              0);
  m_xmax       = config   ("xmax",         100000);
  m_ymin       = config   ("ymin",              0);
  m_ymax       = config   ("ymax",         100000);
  m_event      = config   ("testEvent",         0);
  m_filter     = config   ("filterIsOn",     true);
  m_source     = m_src;
}

//--------------
// Destructor --
//--------------
ImgPeakFinder::~ImgPeakFinder ()
{
}

/// Method which is called once at the beginning of the job
void 
ImgPeakFinder::beginJob(Event& evt, Env& env)
{
  m_time = new TimeInterval();
}

/// Method which is called at the beginning of the run
void 
ImgPeakFinder::beginRun(Event& evt, Env& env)
{
  printInputParameters();
}

/// Method which is called at the beginning of the calibration cycle
void 
ImgPeakFinder::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ImgPeakFinder::event(Event& evt, Env& env)
{
  m_time -> startTimeOnce();

  ++ m_count;
  if ( !m_filter )            { ++ m_selected; return; } // If the filter is OFF then event is selected

  if ( getAndProcImage(evt) ) { ++ m_selected; return; } // if event is selected
  else                        { skip();        return; } // if event is discarded
}
  
/// Method which is called at the end of the calibration cycle
void 
ImgPeakFinder::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
ImgPeakFinder::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
ImgPeakFinder::endJob(Event& evt, Env& env)
{
  m_time -> stopTime(m_count);
  MsgLog(name(), info, "Number of selected events = " << m_selected << " of total " << m_count);
}

//--------------------
//--------------------
//--------------------
//--------------------

/// Print input parameters
void 
ImgPeakFinder::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n source     : "     << m_src
	<< "\n key        : "     << m_key      
	<< "\n peaksKey   : "     << m_peaksKey      
	<< "\n thr_low    : "     << m_thr_low
	<< "\n thr_high   : "     << m_thr_high
	<< "\n sigma      : "     << m_sigma
	<< "\n nsm        : "     << m_nsm
	<< "\n npeak      : "     << m_npeak
	<< "\n xmin       : "     << m_xmin     
	<< "\n xmax       : "     << m_xmax     
	<< "\n ymin       : "     << m_ymin     
	<< "\n ymax       : "     << m_ymax     
	<< "\n event      : "     << m_event     
	<< "\n filterIsOn : "     << m_filter;   
  }
}

//--------------------

bool
ImgPeakFinder::getAndProcImage(Event& evt)
{
  //MsgLog(name(), info, "::getAndProcImage(...)");

  shared_ptr< CSPadPixCoords::Image2D<double> > img2d = evt.get(m_src, m_key, &m_actualSrc); 
  if (img2d.get()) {
    MsgLog(name(), info, "::procImage(...): Get image as Image2D<double>");
    m_img2d = img2d.get();
    const unsigned shape[] = {m_img2d->getNRows(), m_img2d->getNCols()};
    m_ndarr = new ndarray<double,2>(m_img2d->data(), shape);
    return procImage();
  }

  shared_ptr< ndarray<double,2> > img = evt.get(m_src, m_key, &m_actualSrc);
  if (img.get()) {
    MsgLog(name(), info, "::procImage(...): Get image as ndarray<double,2>");
    m_img2d = new CSPadPixCoords::Image2D<double>(img->data(), img->shape()[0], img->shape()[1]);
    m_ndarr = img.get();
    return procImage();
  }

  shared_ptr<Psana::Camera::FrameV1> frmData = evt.get(m_source);
  if (frmData.get()) {

    //unsigned h      = frmData->height();
    //unsigned w      = frmData->width();
    int offset = frmData->offset();

    //m_data = new double [h*w];
    double *p_data = &m_data_arr[0];
    unsigned ind = 0;

      const ndarray<uint8_t, 2>& data8 = frmData->data8();
      if (not data8.empty()) {

	const unsigned *shape = data8.shape();
	ndarray<uint8_t, 2>::const_iterator cit;
	for(cit=data8.begin(); cit!=data8.end(); cit++) { p_data[ind++] = double(int(*cit) - offset); }

        m_ndarr = new ndarray<double,2>(p_data, shape);
        m_img2d = new CSPadPixCoords::Image2D<double>(p_data, shape[0], shape[1]);
        return procImage();
      }

      const ndarray<uint16_t, 2>& data16 = frmData->data16();
      if (not data16.empty()) {

	const unsigned *shape = data16.shape();
	ndarray<uint16_t, 2>::const_iterator cit;
	// This loop consumes ~5 ms/event for Opal1000 camera with 1024x1024 image size 
	for(cit=data16.begin(); cit!=data16.end(); cit++) { p_data[ind++] = double(*cit) - offset; }

        m_ndarr = new ndarray<double,2>(p_data, shape);
        m_img2d = new CSPadPixCoords::Image2D<double>(p_data, shape[0], shape[1]);
        return procImage();
      }
  }

    return false; // if the image object is not found in evt
}

//--------------------

/// Use input parameters and image dimensions and set the window range
void 
ImgPeakFinder::setWindowRange()
{
    static unsigned entrance_counter = 0;
    if( entrance_counter > 0 ) return;
        entrance_counter ++;
  
    m_nrows  = m_img2d -> getNRows();
    m_ncols  = m_img2d -> getNCols();

    m_rowmin = (size_t) min(m_ymin, m_ymax); 
    m_rowmax = (size_t) max(m_ymin, m_ymax);
    m_colmin = (size_t) min(m_xmin, m_xmax);
    m_colmax = (size_t) max(m_xmin, m_xmax);    

    if (m_rowmin < MARGIN           ) m_rowmin = MARGIN;
    if (m_rowmax < MARGIN1          ) m_rowmax = MARGIN1;
    if (m_rowmin >= m_nrows-MARGIN1 ) m_rowmin = m_nrows-MARGIN1;
    if (m_rowmax >= m_nrows-MARGIN  ) m_rowmax = m_nrows-MARGIN;

    if (m_colmin < MARGIN           ) m_colmin = MARGIN;
    if (m_colmax < MARGIN1          ) m_colmax = MARGIN1;
    if (m_colmin >= m_ncols-MARGIN1 ) m_colmin = m_ncols-MARGIN1;
    if (m_colmax >= m_ncols-MARGIN  ) m_colmax = m_ncols-MARGIN;

    evaluateWeights();
}

//--------------------

bool
ImgPeakFinder::procImage()
{
    setWindowRange();
    saveImageInFile0();
    initImage();         // non-zero pixels only in window above the lower threshold
    saveImageInFile1();
    smearImage();        // convolution with Gaussian
    //smearImage();      // convolution with Gaussian
    saveImageInFile2();

    //findPeaks();       // above higher threshold as a maximal in the center of 3x3 or 5x5 
    return true;
}

//--------------------

void 
ImgPeakFinder::saveImageInFile0()
{
  if(m_count != m_event ) return;
  m_img2d -> saveImageInFile("image0_ev" + stringEventN() + ".txt",0);
}

//--------------------

void 
ImgPeakFinder::saveImageInFile1()
{
  if(m_count != m_event ) return;
  m_work2d -> saveImageInFile("image1_ev" + stringEventN() + ".txt",0);
}

//--------------------

void 
ImgPeakFinder::saveImageInFile2()
{
  if(m_count != m_event ) return;
  m_work2d -> saveImageInFile("image2_ev" + stringEventN() + ".txt",0);
}

//--------------------

string 
ImgPeakFinder::stringEventN()
{
  stringstream ssEvNum; ssEvNum << setw(6) << setfill('0') << m_count;
  return ssEvNum.str();
}

//--------------------
// non-zero pixels only in window above the lower threshold
void 
ImgPeakFinder::initImage()
{
  double   val=0;
  unsigned ind=0;
  const double *p_data = m_img2d->data();

  for (size_t r = m_rowmin; r < m_rowmax; r++) {
    for (size_t c = m_colmin; c < m_colmax; c++) {
  	    ind = r*m_ncols + c;
          val = p_data[ind];
  	    m_work_arr[ind] = (val > m_thr_low) ? val : 0;
    }
  }
  m_work2d = new CSPadPixCoords::Image2D<double>(&m_work_arr[0], m_nrows, m_ncols);
}

//--------------------
// Smearing of the image, one pass
void 
ImgPeakFinder::smearImage()
{
  unsigned ind=0;
  const double *p_work = m_work2d->data();

  for (size_t r = m_rowmin; r < m_rowmax; r++) {
    for (size_t c = m_colmin; c < m_colmax; c++) {
  	    ind = r*m_ncols + c;
  	    if (p_work[ind]>0) m_work_arr[ind] = smearPixAmp(r,c);
    }
  }
}

//--------------------
// Smearing of a single-pixel ampliude
double 
ImgPeakFinder::smearPixAmp(size_t r0, size_t c0)
{
  double sum_aw = 0;
  double sum_w  = 0;
  double     w  = 0;

  unsigned ind=0;
  const double *p_data = m_img2d->data();

  for (int dr = -m_nsm; dr <= m_nsm; dr++) { size_t r = r0 + dr;
   for (int dc = -m_nsm; dc <= m_nsm; dc++) { size_t c = c0 + dc;

      ind = r*m_ncols + c;
      w = weight(dr,dc);
      sum_w  += w;
      sum_aw += p_data[ind] * w;

      //cout << "dr, dc, ind, w=" << dr << " " << dc << " " << ind << " " << w << endl;

   }
  }
  return sum_aw / sum_w;
}

//--------------------
// Get smearing weight
double 
ImgPeakFinder::weight(int dr, int dc)
{
  return m_weights[abs(dr)][abs(dc)];
}

//--------------------
// Define smearing weighting matrix
void 
ImgPeakFinder::evaluateWeights()
{
  WithMsgLog(name(), info, log) { log << "::evaluateWeights():";
    for (int r = 0; r <= m_nsm; r++) {
          log << "\n   row=" << r << ":     "; 
      for (int c = 0; c <= m_nsm; c++) {
        m_weights[r][c] = exp( -0.5*(r*r+c*c) / (m_sigma*m_sigma) );
          log << "   " << m_weights[r][c];       
      }
    }
  } // WithMsgLog 
}

//--------------------
//--------------------
//--------------------
//--------------------
//--------------------
//--------------------
//--------------------
//--------------------


} // namespace ImgAlgos
