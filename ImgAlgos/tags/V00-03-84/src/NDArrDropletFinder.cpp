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
  , m_key_sme()
  , m_thr_low()
  , m_thr_high()
  , m_sigma()
  , m_nsm()
  , m_rpeak()
  , m_low_value()
  , m_windows()  
  , m_fname_mask()
  , m_mask_val()
  , m_ofname_pref()
  , m_print_bits()
  , m_count_evt(0)
  , m_count_get(0)
  , m_count_msg(0)
  , m_count_sel(0)
  , m_mask_data(0)
{
  // get the values from configuration or use defaults
  m_source        = configSrc("source",  "DetInfo()");
  m_key           = configStr("key",              "");
  m_key_out       = configStr("key_droplets",     "");
  m_key_sme       = configStr("key_smeared",      "");
  m_thr_low       = config   ("threshold_low",    10);
  m_thr_high      = config   ("threshold_high",  100);
  m_sigma         = config   ("sigma",           1.5);
  m_nsm           = config   ("smear_radius",      3);
  m_rpeak         = config   ("peak_radius",       3);
  m_low_value     = config   ("low_value",         0);
  m_windows       = configStr("windows",          "");
  m_fname_mask    = configStr("mask",             "");
  m_mask_val      = config   ("masked_value",     0.);
  m_ofname_pref   = configStr("fname_prefix",     "");
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
  if (m_print_bits & 1) {
    printInputPars();
    print_windows();
  }
  m_time = new TimeInterval();
}

/// Method which is called at the beginning of the run
void 
NDArrDropletFinder::beginRun(Event& evt, Env& env)
{
  m_str_exp = stringExperiment(env);
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
  if (m_print_bits & 2) {
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
	 << "\n key_sme       : " << m_key_sme
	 << "\n thr_low       : " << m_thr_low
	 << "\n thr_high      : " << m_thr_high
	 << "\n sigma         : " << m_sigma
	 << "\n rsm           : " << m_nsm
	 << "\n npeak         : " << m_rpeak
	 << "\n low_value     : " << m_low_value
	 << "\n windows       : " << m_windows
	 << "\n fname_mask    : " << m_fname_mask
	 << "\n mask_val      : " << m_mask_val
         << "\n ofname_pref   : " << m_ofname_pref
         << "\n print_bits    : " << m_print_bits
	)
}

//--------------------

std::string 
NDArrDropletFinder::getCommonFileName(Event& evt)
{
  std::string fname; 
  fname = m_ofname_pref
        + "-"   + m_str_exp
        + "-r"  + stringRunNumber(evt) 
        + "-e"  + stringFromUint(m_count_evt);
  //+ "-"   + stringTimeStamp(evt) 
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

  if (m_windows.empty()) {
    if (m_print_bits) MsgLog(name(), warning, "The list of windows is empty. " 
                                              << "All segments will be processed");
    // throw std::runtime_error("Check NDArrDropletFinder parameters in the configuration file!");
    return;
  }

  if (m_print_bits & 256) MsgLog(name(), info, "Parse window parameters:");

  unsigned ind = 0;
  while (ss >> s) {
    if (!s.empty()) v[ind++] = atoi(s.c_str());
    if (ind < nvals) continue;
    ind = 0;
    WINDOW win = {v[0], v[1], v[2], v[3], v[4]};
    v_windows.push_back(win);

    if (m_print_bits & 256) MsgLog( name(), info, "Window for"
                                   << "     seg:" << std::setw(3) << v[0]
            			   << "  rowmin:" << std::setw(6) << v[1] 
            			   << "  rowmax:" << std::setw(6) << v[2] 
            			   << "  colmin:" << std::setw(6) << v[3] 
				   << "  colmax:" << std::setw(6) << v[4] );
  }

  if (v_windows.empty() && m_print_bits) { MsgLog(name(), warning, "Vector of window parameters is empty." 
                                                                   << " All segments will be processed.");
  }
  else if (m_print_bits & 256) MsgLog(name(), info, "Number of specified windows: " 
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
  if (++m_count_msg < 11) {
    MsgLog(name(), info, "method:"<< std::setw(10) << add_msg << " input ndarray is not available in the event:" 
                         << m_count_evt << " for source:\"" << m_source << "\"  key:\"" << m_key << "\"");
    if (m_count_msg == 10) MsgLog(name(), warning, "STOP WARNINGS for source:\"" << m_source 
                                                   << "\"  key:\"" << m_key << "\"");    
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

  if (m_print_bits) printWarningMsg("initProc");
}

//--------------------

void 
NDArrDropletFinder::procEvent(Event& evt, Env& env)
{
  if (! m_count_get) initProc(evt, env);
  if (! m_count_get) return;

  if      ( m_dtype == INT16  && procEventForType<int16_t > (evt)) return;
  else if ( m_dtype == INT    && procEventForType<int     > (evt)) return;
  else if ( m_dtype == FLOAT  && procEventForType<float   > (evt)) return;
  else if ( m_dtype == DOUBLE && procEventForType<double  > (evt)) return;
  else if ( m_dtype == UINT16 && procEventForType<uint16_t> (evt)) return;
  else if ( m_dtype == UINT8  && procEventForType<uint8_t > (evt)) return;

  if (m_print_bits) printWarningMsg("procEvent");
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

void
NDArrDropletFinder::appendVectorOfDroplets(const std::vector<AlgDroplet::Droplet>& v)
{
  for( vector<AlgDroplet::Droplet>::const_iterator it  = v.begin();
                                                   it != v.end(); it++ ) {
    v_droplets.push_back(*it); 
  }
}

//--------------------

void
NDArrDropletFinder::saveDropletsInEvent(Event& evt)
{	
  if (m_print_bits & 4) MsgLog(name(), info, "Save array of " << v_droplets.size() << " droplets in the event");

  size_t ndroplets = v_droplets.size();
  ndarray<droplet_t,2> nda = make_ndarray<droplet_t>(ndroplets, 6);

  int i=0;
  for( vector<AlgDroplet::Droplet>::iterator it  = v_droplets.begin();
                                             it != v_droplets.end(); it++, i++ ) {
    droplet_t* p = &nda[i][0];
    p[0]= droplet_t(it->seg);
    p[1]= droplet_t(it->row);
    p[2]= droplet_t(it->col);
    p[3]= droplet_t(it->ampmax);
    p[4]= droplet_t(it->amptot);
    p[5]= droplet_t(it->npix);
  }

  if (m_print_bits & 8) MsgLog(name(), info, "Save in the event store ndarray with peaks:\n" << nda);

  save2DArrayInEvent<droplet_t>(evt, m_src, m_key_out, nda);
}

//--------------------
//void 
//NDArrDropletFinder::saveNDArrInFile(Event& evt)
//{
//  std::string fname = getCommonFileName(evt) + "-smeared.txt"; 
//  MsgLog( name(), info, "Save ndarray in file:" << fname.data() );
//  //m_work2d -> saveImageInFile(fname,0);
//}

//--------------------
// Save peak vector info in the file
void 
NDArrDropletFinder::saveDropletsInFile(Event& evt)
{
  string fname; fname = getCommonFileName(evt) + "-peaks.txt";
  if (m_print_bits & 16) MsgLog( name(), info, "Save the peak info in file:" << fname.data() );

  ofstream file; 
  file.open(fname.c_str(),ios_base::out);

  for( std::vector<AlgDroplet::Droplet>::iterator itv  = v_droplets.begin();
                                                  itv != v_droplets.end(); itv++ ) {

    file  << std::setw(8) << std::left << itv->seg << "  "
          << std::setw(8) << itv->row      << "  "
          << std::setw(8) << itv->col      << "  "
          << std::setw(8) << itv->ampmax   << "  "
          << std::setw(8) << itv->amptot   << "  "
          << std::setw(8) << itv->npix     << endl; 
  }

  file.close();
}


//--------------------
//--------------------

} // namespace ImgAlgos

//---------EOF--------
