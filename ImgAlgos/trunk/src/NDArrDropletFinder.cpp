//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class NDArrDropletFinder...
//
// Author:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

#include "ImgAlgos/NDArrDropletFinder.h"
#include "ImgAlgos/GlobalMethods.h"

#include <math.h> // for exp
#include <fstream> // for ofstream

#include "psddl_psana/camera.ddl.h"
#include "PSEvt/EventId.h"
//#include "PSTime/Time.h"

#include <iomanip>   // for setw, setfill
#include <sstream>   // for stringstream
#include <iostream>  // for setf
#include <algorithm> // for fill_n

// This declares this class as psana module
using namespace std;
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(NDArrDropletFinder)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
NDArrDropletFinder::NDArrDropletFinder (const std::string& name)
  : Module(name)
  , m_source()
  , m_key()
  , m_key_out()
  , m_thr_low()
  , m_thr_high()
  , m_sigma()
  , m_nsm()
  , m_rpeak()
  , m_windows()  
  , m_event()
  , m_print_bits()
  , m_count_evt(0)
  , m_count_get(0)
  , m_count_msg(0)
  , m_count_sel(0)
{
  // get the values from configuration or use defaults
  m_source        = configSrc("source",  "DetInfo()");
  m_key           = configStr("key",              "");
  m_key_out       = configStr("key_droplets",     "");
  m_thr_low       = config   ("threshold_low",    10);
  m_thr_high      = config   ("threshold_high",  100);
  m_sigma         = config   ("sigma",           1.5);
  m_nsm           = config   ("smear_radius",      3);
  m_rpeak         = config   ("peak_radius",       3);
  m_windows       = configStr("windows",          "");
  m_event         = config   ("testEvent",         0);
  m_print_bits    = config   ("print_bits",        0);

  //std::fill_n(&m_data_arr[0], int(MAX_IMG_SIZE), double(0));
  //std::fill_n(&m_work_arr[0], int(MAX_IMG_SIZE), double(0));

  parse_windows_pars();
}

//--------------
// Destructor --
//--------------
NDArrDropletFinder::~NDArrDropletFinder ()
{
      std::vector<AlgSmearing*>::iterator itsm = v_algsm.begin();
      std::vector<AlgDroplet*>::iterator  itdf = v_algdf.begin();
      for ( ; itdf != v_algdf.end(); ++itdf, ++itsm) { 
        delete (*itdf);
        delete (*itsm);
      }
      v_algdf.clear();
      v_algsm.clear();
}

/// Method which is called once at the beginning of the job
void 
NDArrDropletFinder::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) {
    printInputPars();
    print_windows();
  }

  m_time = new TimeInterval();
}

/// Method which is called at the beginning of the run
void 
NDArrDropletFinder::beginRun(Event& evt, Env& env)
{
  //  evaluateWeights();
  //  if( m_print_bits & 1 ) printWeights();
}

/// Method which is called at the beginning of the calibration cycle
void 
NDArrDropletFinder::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
NDArrDropletFinder::event(Event& evt, Env& env)
{
  m_time -> startTimeOnce();
  ++ m_count_evt;

  procEvent(evt, env);

  //if ( !m_finderIsOn )        { ++  m_count_sel; return; } // If the filter is OFF then event is selected
  //if ( getAndProcImage(evt) ) { ++  m_count_sel; return; } // if event is selected
  //else                        { skip();          return; } // if event is discarded
}
  
/// Method which is called at the end of the calibration cycle
void 
NDArrDropletFinder::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
NDArrDropletFinder::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
NDArrDropletFinder::endJob(Event& evt, Env& env)
{
  if( m_print_bits & 2 ) {
    MsgLog(name(), info, "Number of events with found data = " << m_count_sel << " of total " << m_count_evt
	   << " for source:" << m_source << " and key:" << m_key);
    m_time -> stopTime(m_count_evt);
  }
}

//--------------------
//--------------------
//--------------------
//--------------------

// Print input parameters
void 
NDArrDropletFinder::printInputPars()
{
  MsgLog(name(), info, "\n Input parameters:"
         << "\n source        : " << m_source
	 << "\n key           : " << m_key      
	 << "\n key_out       : " << m_key_out
	 << "\n thr_low       : " << m_thr_low
	 << "\n thr_high      : " << m_thr_high
	 << "\n sigma         : " << m_sigma
	 << "\n rsm           : " << m_nsm
	 << "\n npeak         : " << m_rpeak
	 << "\n windows       : " << m_windows
	 << "\n event         : " << m_event     
	 << "\n print_bits    : " << m_print_bits;
	)
}

//--------------------

std::string 
NDArrDropletFinder::getCommonFileName(Event& evt)
{
  std::string fname; 
  fname = "nda-r" + stringRunNumber(evt) 
        + "-"     + stringTimeStamp(evt) 
        + "-ev"   + stringFromUint(m_count_evt);
  return fname;
}

//--------------------

void
NDArrDropletFinder::parse_windows_pars()
{
  v_windows.reserve(N_WINDOWS_BLK);

  std::stringstream ss(m_windows);
  std::string s;  

  const size_t nvals = 5;
  int v[nvals];

  if(m_windows.empty()) {
    MsgLog(name(), warning, "The list of windows is empty. " 
                         << "All segments will be processed");
    // throw std::runtime_error("Check CSPad2x2NDArrReshape parameters in the configuration file!");
    return;
  }

  if( m_print_bits & 4 ) MsgLog(name(), info, "Parse window parameters:");

  unsigned ind = 0;
  while (ss >> s) {
    if (!s.empty()) v[ind++] = atoi(s.c_str());
    if (ind < nvals) continue;
    ind = 0;
    WINDOW win = {v[0], v[1], v[2], v[3], v[4]};
    v_windows.push_back(win);

    if( m_print_bits & 4 ) MsgLog( name(), info, "Window for"
                                   << "     seg:" << std::setw(3) << v[0]
            			   << "  rowmin:" << std::setw(6) << v[1] 
            			   << "  rowmax:" << std::setw(6) << v[2] 
            			   << "  colmin:" << std::setw(6) << v[3] 
				   << "  colmax:" << std::setw(6) << v[4] );
  }

  if (v_windows.empty()) { MsgLog(name(), warning, "Vector of window parameters is empty." 
                                  << " Entire segments will be processed.");
  }
  else if( m_print_bits & 4 ) MsgLog(name(), info, "Number of specified windows: " 
                                     << v_windows.size());
}

//--------------------

void
NDArrDropletFinder::print_windows()
{
      std::stringstream ss; ss << "Vector of windows of size: " << v_windows.size();

      std::vector<WINDOW>::iterator it  = v_windows.begin();
      for ( ; it != v_windows.end(); ++it) 
        ss  << "\n   seg:" << std::setw(8) << std::left << it->seg
            << "  rowmin:" << std::setw(8) << it->rowmin 
            << "  rowmax:" << std::setw(8) << it->rowmax 
            << "  colmin:" << std::setw(8) << it->colmin 
            << "  colmax:" << std::setw(8) << it->colmax;
      ss  << '\n';

      MsgLog(name(), info, ss.str());
}

//--------------------

void 
NDArrDropletFinder::printWarningMsg(const std::string& add_msg)
{
  if (++m_count_msg < 11 && m_print_bits) {
    MsgLog(name(), info, "method:"<< std::setw(10) << add_msg << " input ndarray is not available in the event:" << m_count_evt 
                         << " for source:\"" << m_source << "\"  key:\"" << m_key << "\"");
    if (m_count_msg == 10) MsgLog(name(), warning, "STOP WARNINGS for source:\"" << m_source << "\"  key:\"" << m_key << "\"");    
  }
}

//--------------------

void
NDArrDropletFinder::initProc(Event& evt, Env& env)
{
  if      ( initProcForType<int16_t > (evt) ) {m_dtype = INT16;  return;}
  else if ( initProcForType<int     > (evt) ) {m_dtype = INT;    return;}
  else if ( initProcForType<float   > (evt) ) {m_dtype = FLOAT;  return;}
  else if ( initProcForType<double  > (evt) ) {m_dtype = DOUBLE; return;}
  else if ( initProcForType<uint16_t> (evt) ) {m_dtype = UINT16; return;}
  else if ( initProcForType<uint8_t > (evt) ) {m_dtype = UINT8;  return;}

  printWarningMsg("initProc");
}

//--------------------

void 
NDArrDropletFinder::procEvent(Event& evt, Env& env)
{
  if ( ! m_count_get ) initProc(evt, env);
  if ( ! m_count_get ) return;

  if      ( m_dtype == INT16  && procEventForType<int16_t > (evt)) return;
  else if ( m_dtype == INT    && procEventForType<int     > (evt)) return;
  else if ( m_dtype == FLOAT  && procEventForType<float   > (evt)) return;
  else if ( m_dtype == DOUBLE && procEventForType<double  > (evt)) return;
  else if ( m_dtype == UINT16 && procEventForType<uint16_t> (evt)) return;
  else if ( m_dtype == UINT8  && procEventForType<uint8_t > (evt)) return;

  printWarningMsg("procEvent");
}

//--------------------

void 
NDArrDropletFinder::printFoundNdarray()
{
          MsgLog(name(), info, "printFoundNdarray(): found ndarray with NDim:" << m_ndim
                   << "  dtype:"   << strDataType(m_dtype)
                   << "  isconst:" << m_isconst
		 );
}

//--------------------
//--------------------

/*
bool
NDArrDropletFinder::getAndProcImage(Event& evt)
{
  //MsgLog(name(), info, "::getAndProcImage(...)");

  shared_ptr< CSPadPixCoords::Image2D<double> > img2d = evt.get(m_source, m_key, &m_src); 
  if (img2d.get()) {
    if( m_print_bits & 16 ) MsgLog(name(), info, "getAndProcImage(...): Get image as Image2D<double>");
    m_img2d = img2d.get();
    const unsigned shape[] = {m_img2d->getNRows(), m_img2d->getNCols()};
    m_ndarr = new ndarray<const double,2>(m_img2d->data(), shape);
    return procImage(evt);
  }

  shared_ptr< ndarray<const double,2> > img = evt.get(m_source, m_key, &m_src);
  if (img.get()) {
    if( m_print_bits & 16 ) MsgLog(name(), info, "getAndProcImage(...): Get image as ndarray<double,2>");
    m_img2d = new CSPadPixCoords::Image2D<double>(img->data(), img->shape()[0], img->shape()[1]);
    m_ndarr = img.get();
    return procImage(evt);
  }

  shared_ptr<Psana::Camera::FrameV1> frmData = evt.get(m_source, "", &m_src);
  if (frmData.get()) {

    //unsigned h      = frmData->height();
    //unsigned w      = frmData->width();
    int offset = frmData->offset();

    //m_data = new double [h*w];
    double *p_data = &m_data_arr[0];
    unsigned ind = 0;

      const ndarray<const uint8_t, 2>& data8 = frmData->data8();
      if (not data8.empty()) {
        if( m_print_bits & 16 ) MsgLog(name(), info, "getAndProcImage(...): Get image as ndarray<const uint8_t,2>");
	const unsigned *shape = data8.shape();
	ndarray<const uint8_t, 2>::iterator cit;
	for(cit=data8.begin(); cit!=data8.end(); cit++) { p_data[ind++] = double(int(*cit) - offset); }

        m_ndarr = new ndarray<const double,2>(p_data, shape);
        m_img2d = new CSPadPixCoords::Image2D<double>(p_data, shape[0], shape[1]);
        return procImage(evt);
      }

      const ndarray<const uint16_t, 2>& data16 = frmData->data16();
      if (not data16.empty()) {
        if( m_print_bits & 16 ) MsgLog(name(), info, "getAndProcImage(...): Get image as ndarray<const uint16_t,2>");
	const unsigned *shape = data16.shape();
	ndarray<const uint16_t, 2>::iterator cit;
	// This loop consumes ~5 ms/event for Opal1000 camera with 1024x1024 image size 
	for(cit=data16.begin(); cit!=data16.end(); cit++) { p_data[ind++] = double(*cit) - offset; }

        m_ndarr = new ndarray<const double,2>(p_data, shape);
        m_img2d = new CSPadPixCoords::Image2D<double>(p_data, shape[0], shape[1]);

        return procImage(evt);
      }
  }
    return false; // if the image object is not found in evt
}

*/

//--------------------

/*
void 
NDArrDropletFinder::printWindowRange()
{
    MsgLog(name(), info, "Window range: m_colmin=" << m_colmin
                                   << " m_colmax=" << m_colmax
                                   << " m_rowmin=" << m_rowmin
                                   << " m_rowmax=" << m_rowmax );
}
*/

//--------------------

 /*
bool
NDArrDropletFinder::procImage(Event& evt)
{
                     v_peaks.clear();
                     v_peaks.reserve(100);
                     setWindowRange();
                     saveImageInFile0(evt);
    if(m_sigma != 0) smearImage();        // convolution with Gaussian

    else             initImage();         // non-zero pixels only in window above the lower threshold
                     saveImageInFile1(evt);
		     findPeaks(evt);      // above higher threshold as a maximal in the center of 3x3 or 5x5 
		     savePeaksInEvent(evt);
		     savePeaksInEventAsNDArr(evt);
		     savePeaksInFile(evt);
    return true;
}

 */

//--------------------

  /*
void 
NDArrDropletFinder::saveImageInFile0(Event& evt)
{
  if(m_count != m_event ) return;
  m_img2d -> saveImageInFile(getCommonFileName(evt) + ".txt",0);
}
  */
//--------------------
/*
void 
NDArrDropletFinder::saveImageInFile1(Event& evt)
{
  if(m_count != m_event ) return;
  m_work2d -> saveImageInFile(getCommonFileName(evt) + "-smeared.txt",0);
}
  */
//--------------------
/*
void 
NDArrDropletFinder::saveImageInFile2(Event& evt)
{
  if(m_count != m_event ) return;
  m_work2d -> saveImageInFile(getCommonFileName(evt) + "-2.txt",0);
}

*/

//--------------------
// Save peak vector in the event as ndarray
/*
void 
NDArrDropletFinder::savePeaksInEventAsNDArr(Event& evt)
{
  if(m_key_out.empty()) return;

  ndarray<float, 2> peaks_nda = make_ndarray<float>(int(v_peaks.size()), 5);

  int i=-1;
  for(vector<Peak>::const_iterator itv  = v_peaks.begin();
                                   itv != v_peaks.end(); itv++) {
    i++;
    peaks_nda[i][0] = float(itv->x);
    peaks_nda[i][1] = float(itv->y);
    peaks_nda[i][2] = float(itv->ampmax);
    peaks_nda[i][3] = float(itv->amptot);
    peaks_nda[i][4] = float(itv->npix);
  }

  save2DArrayInEvent<float>(evt, m_src, m_key_out, peaks_nda);
}
*/

//--------------------
// Save peak vector info in the file for test event
 /*
void 
NDArrDropletFinder::savePeaksInFile(Event& evt)
{
  if(m_count != m_event ) return;

  string fname; fname = getCommonFileName(evt) + "-peaks.txt";
  MsgLog( name(), info, "Save the peak info in file:" << fname.data() );

  ofstream file; 
  file.open(fname.c_str(),ios_base::out);

  for( std::vector<Peak>::iterator itv  = v_peaks.begin();
                                   itv != v_peaks.end(); itv++ ) {

    if( m_print_bits & 16 ) printPeakInfo(*itv);

    file << itv->x        << "  "
         << itv->y	  << "  "
         << itv->ampmax	  << "  "
         << itv->amptot	  << "  "
         << itv->npix     << endl; 
  }

  file.close();
}
 */

//--------------------

} // namespace ImgAlgos

//---------EOF--------
