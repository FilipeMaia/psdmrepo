//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class NDArrAverage...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/NDArrAverage.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <fstream>
#include <cmath>
#include <sstream>   // for stringstream
#include <boost/filesystem.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
//#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace Psana;
using namespace ImgAlgos;

PSANA_MODULE_FACTORY(NDArrAverage)

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
NDArrAverage::NDArrAverage (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key()
  , m_sumFile()
  , m_aveFile()
  , m_rmsFile()
  , m_mskFile()
  , m_hotFile()
  , m_maxFile()
  , m_file_type()
  , m_thr_rms()
  , m_thr_min()
  , m_thr_max()
  , m_print_bits()
  , m_count(0)
  , m_count_ev(0)
  , m_nev_stage1()
  , m_nev_stage2()
  , m_gate_width1()
  , m_gate_width2()
{
  // get the values from configuration or use defaults
  m_str_src     = configSrc("source",      "DetInfo(:Cspad)");
  m_key         = configStr("key",         "");
  m_sumFile     = configStr("sumfile",     "");
  m_aveFile     = configStr("avefile",     "");
  m_rmsFile     = configStr("rmsfile",     "");
  m_mskFile     = configStr("maskfile",    "");
  m_hotFile     = configStr("hotpixfile",  "");
  m_maxFile     = configStr("maxfile",     "");
  m_file_type   = configStr("ftype",    "txt");
  m_thr_rms     = config("thr_rms_ADU",   10000.);
  m_thr_min     = config("thr_min_ADU", -100000.);
  m_thr_max     = config("thr_max_ADU",  100000.);
  m_nev_stage1  = config("evts_stage1",  1000000);
  m_nev_stage2  = config("evts_stage2",        0);
  m_gate_width1 = config("gate_width1",        0); 
  m_gate_width2 = config("gate_width2",        0); 
  m_print_bits  = config("print_bits",         0);
 
  m_do_sum  = (m_sumFile.empty()) ? false : true;
  m_do_ave  = (m_aveFile.empty()) ? false : true;
  m_do_rms  = (m_rmsFile.empty()) ? false : true;
  m_do_msk  = (m_mskFile.empty()) ? false : true;
  m_do_hot  = (m_hotFile.empty()) ? false : true;
  m_do_max  = (m_maxFile.empty()) ? false : true;

  // If all file names are empty - save average and rms with default names
  if (    !m_do_msk 
       && !m_do_hot 
       && !m_do_sum 
       && !m_do_ave
       && !m_do_rms ) {

    m_aveFile = "arr-ave";
    m_rmsFile = "arr-rms";
    m_do_ave  = true;
    m_do_rms  = true;
  }

  m_ndarr_pars = 0;
  setFileMode();

  if( m_print_bits & 1 ) printInputParameters();
 }

//--------------------

// Print input parameters
void 
NDArrAverage::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n source     : " << m_str_src
        << "\n key        : " << m_key      
        << "\n m_sumFile  : " << m_sumFile    
        << "\n m_aveFile  : " << m_aveFile    
        << "\n m_rmsFile  : " << m_rmsFile    
        << "\n m_mskFile  : " << m_mskFile    
        << "\n m_hotFile  : " << m_hotFile    
        << "\n m_maxFile  : " << m_maxFile    
        << "\n m_ftype    : " << m_file_type
        << "\n m_thr_rms  : " << m_thr_rms  
        << "\n m_thr_min  : " << m_thr_min  
        << "\n m_thr_max  : " << m_thr_max
        << "\n m_do_sum   : " << m_do_sum
        << "\n m_do_ave   : " << m_do_ave
        << "\n m_do_rms   : " << m_do_rms
        << "\n m_do_msk   : " << m_do_msk
        << "\n m_do_hot   : " << m_do_hot
        << "\n m_do_max   : " << m_do_max
        << "\n print_bits : " << m_print_bits
        << "\n evts_stage1: " << m_nev_stage1   
        << "\n evts_stage2: " << m_nev_stage2  
        << "\n gate_width1: " << m_gate_width1 
        << "\n gate_width2: " << m_gate_width2 
        << "\n";     
  }
}

//--------------------

void
NDArrAverage::setFileMode()
{
  m_file_mode = TEXT;
  if (m_file_type == "bin")     { m_file_mode = BINARY;    return; }
  if (m_file_type == "txt")     { m_file_mode = TEXT;      return; }
  if (m_file_type == "metatxt") { m_file_mode = METADTEXT; return; }

  const std::string msg = "The output file type: " + m_file_type + " is not recognized. Known types are: bin, txt, metatxt. Will use txt";
  MsgLog(name(), warning, msg);
  //MsgLogRoot(error, msg);
  //throw std::runtime_error(msg);
}



//--------------
// Destructor --
//--------------
NDArrAverage::~NDArrAverage ()
{
}

/// Method which is called once at the beginning of the job
void 
NDArrAverage::beginJob(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the run
void 
NDArrAverage::beginRun(Event& evt, Env& env)
{
  m_str_exp     = stringExperiment(env);
  m_str_run_num = stringRunNumber(evt);

  boost::filesystem::path path = m_aveFile;
  if ( path.extension().string() == string(".txt") ) m_fname_ext = "";
  else                                               m_fname_ext = "-" + m_str_exp + "-r" + m_str_run_num + ".dat";

  std::stringstream ss; ss << m_str_src;
  m_str_source = ss.str();
}

/// Method which is called at the beginning of the calibration cycle
void 
NDArrAverage::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
NDArrAverage::event(Event& evt, Env& env)
{
  ++ m_count_ev;
  if ( m_print_bits & 2 ) printEventRecord(evt);
  if (! setCollectionMode(evt)) return;
  if (collectStat(evt)) ++ m_count;
}
  
/// Method which is called at the end of the calibration cycle
void 
NDArrAverage::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
NDArrAverage::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
NDArrAverage::endJob(Event& evt, Env& env)
{
  if (m_count == 0) {
    MsgLog(name(), warning, "ndarray object for src: " << m_str_src << " and key: " << m_key 
                            << " IS NOT FOUND in " << m_count_ev << " events.\nFiles are NOT SAVED!!!");
    return;
  }

  procStatArrays();

  std::vector<std::string> v_com;
  std::string com1;
  com1="PRODUCER   "; v_com.push_back(com1 + "ImgAlgos/NDArrAverage");
  com1="EXPERIMENT "; v_com.push_back(com1 + m_str_exp);
  com1="RUN        "; v_com.push_back(com1 + m_str_run_num);
  com1="SOURCE     "; v_com.push_back(com1 + m_str_source);

  std::vector<std::string> v_com1(v_com); v_com1.push_back("TYPE       Sum of events");
  std::vector<std::string> v_com2(v_com); v_com2.push_back("TYPE       Average");
  std::vector<std::string> v_com3(v_com); v_com3.push_back("TYPE       RMS");
  std::vector<std::string> v_com4(v_com); v_com4.push_back("TYPE       Mask of bad pixels");
  std::vector<std::string> v_com5(v_com); v_com5.push_back("TYPE       Pixel status");
  std::vector<std::string> v_com6(v_com); v_com6.push_back("TYPE       Maximal values");

  if (m_do_sum)  saveNDArrayInFile<double> ( m_sumFile+m_fname_ext, m_sum, m_ndarr_pars, m_print_bits & 16, m_file_mode, v_com1 );
  if (m_do_ave)  saveNDArrayInFile<double> ( m_aveFile+m_fname_ext, m_ave, m_ndarr_pars, m_print_bits & 16, m_file_mode, v_com2 );
  if (m_do_rms)  saveNDArrayInFile<double> ( m_rmsFile+m_fname_ext, m_rms, m_ndarr_pars, m_print_bits & 16, m_file_mode, v_com3 );
  if (m_do_msk)  saveNDArrayInFile<int>    ( m_mskFile+m_fname_ext, m_msk, m_ndarr_pars, m_print_bits & 16, m_file_mode, v_com4 );
  if (m_do_hot)  saveNDArrayInFile<int>    ( m_hotFile+m_fname_ext, m_hot, m_ndarr_pars, m_print_bits & 16, m_file_mode, v_com5 );
  if (m_do_max)  saveNDArrayInFile<double> ( m_maxFile+m_fname_ext, m_max, m_ndarr_pars, m_print_bits & 16, m_file_mode, v_com6 );

  if( m_print_bits & 32 ) printSummaryForParser(evt);
  if( m_print_bits & 64 ) printStatBadPix();
}

//--------------------

/// Check the event counter and deside what to do next accumulate/change mode/etc.
bool 
NDArrAverage::setCollectionMode(Event& evt)
{
  // Set the statistics collection mode without gate
  if (m_count == 0) {
    // shape is not available in beginJob and beginRun, so need to define it in the 1st event

    m_ndarr_pars = new NDArrPars();

    if ( ! defineNDArrPars(evt, m_str_src, m_key, m_ndarr_pars, bool(m_print_bits)) ) return false;

    m_size = m_ndarr_pars -> size();

    if( m_print_bits & 1 ) {
      MsgLog(name(), info, "NDArrPars parameters are found in event: " << m_count_ev << " for source:" << m_str_src << " key:" << m_key);
      m_ndarr_pars -> print();
    }
    m_stat = new unsigned[m_size];
    m_sum  = new double  [m_size];
    m_sum2 = new double  [m_size];
    m_ave  = new double  [m_size];
    m_rms  = new double  [m_size]; 
    m_msk  = new int     [m_size]; 
    m_hot  = new int     [m_size]; 
    m_max  = new double  [m_size]; 
    resetStatArrays();
    m_gate_width = 0;
    if( m_print_bits & 4 ) MsgLog(name(), info, "Stage 0: evt = " << m_count << " Begin to collect statistics without gate.");
    return true;
  }

  // Change the statistics collection mode for gated stage 1
  else if (m_count == m_nev_stage1 ) {
    procStatArrays();
    resetStatArrays();
    m_gate_width = m_gate_width1;
    if( m_print_bits & 4 ) MsgLog(name(), info, "Stage 1: evt = " << m_count << " Begin to collect statistics with gate =" << m_gate_width);
    return true;
  } 

  // Change the statistics collection mode for gated stage 2
  else if (m_count == m_nev_stage1 + m_nev_stage2 ) {
    procStatArrays();
    resetStatArrays();
    m_gate_width = m_gate_width2;
    if( m_print_bits & 4 ) MsgLog(name(), info, "Stage 2: evt = " << m_count << " Begin to collect statistics with gate =" << m_gate_width);
    return true;
  }
  return true;
}

//--------------------

/// Reset arrays for statistics accumulation
void
NDArrAverage::resetStatArrays()
{
  std::fill_n(m_stat, int(m_size), unsigned(0));
  std::fill_n(m_sum,  int(m_size), double(0.));
  std::fill_n(m_sum2, int(m_size), double(0.));
  if (m_do_max) std::fill_n(m_max,  int(m_size), double(0.));
}

//--------------------

/// Collect statistics
bool 
NDArrAverage::collectStat(Event& evt)
{
  DATA_TYPE dtype = m_ndarr_pars->dtype();
  if      (dtype == DOUBLE   && collectStatForType<double>  (evt)) return true;
  else if (dtype == UINT16   && collectStatForType<uint16_t>(evt)) return true;
  else if (dtype == INT      && collectStatForType<int>     (evt)) return true;
  else if (dtype == FLOAT    && collectStatForType<float>   (evt)) return true;
  else if (dtype == UINT8    && collectStatForType<uint8_t> (evt)) return true;
  else if (dtype == SHORT    && collectStatForType<short>   (evt)) return true;
  else if (dtype == INT16    && collectStatForType<int16_t> (evt)) return true;
  else if (dtype == UNSIGNED && collectStatForType<unsigned>(evt)) return true;

  static unsigned m_count_msg=0; m_count_msg ++;
  if (m_count_msg < 11 && m_print_bits) {
    MsgLog(name(), warning, "ndarray object is not available in the event: " << m_count_ev << " for source:" 
                            << m_str_src << " key:" << m_key);
    if (m_count_msg == 10) MsgLog(name(), warning, "STOP PRINT WARNINGS for source:" << m_str_src << " key:" << m_key);
  }
  return false;
}

//--------------------

/// Process accumulated stat arrays and evaluate m_ave(rage) and m_rms arrays
void 
NDArrAverage::procStatArrays()
{
  if( m_print_bits & 8 ) MsgLog(name(), info, "Process statistics for collected total " << m_count 
                                              << " ndarrays found in " << m_count_ev << " events");
  
    for (unsigned i=0; i!=m_size; ++i) {

        double stat = m_stat[i];
        if (stat > 0) {
          double ave = m_sum[i] / stat;
	  m_ave[i] = ave;
          m_rms[i] = std::sqrt(m_sum2[i] / stat - ave*ave);
        } 
        else 
        {
	  m_ave[i] = 0;
	  m_rms[i] = 0;
        }
    }

    // Use default or non-default or default (auto-evaluated) threshold
    double thr_rms = (m_thr_rms>0) ? m_thr_rms : evaluateThresholdOnRMS();

    m_nbadpix = 0;
    if (m_do_msk || m_do_hot) {
      for (unsigned i=0; i!=m_size; ++i) {
 	 bool is_bad_pixel = m_rms[i] >   thr_rms
                          || m_ave[i] < m_thr_min
                          || m_ave[i] > m_thr_max;

         if (is_bad_pixel) { m_msk[i] = 1; ++ m_nbadpix; }
	 else                m_msk[i] = 0;

         m_hot[i] = 0;                            // good pixel
	 if (m_rms[i] >   thr_rms) m_hot[i]  = 1; // hot pixel
	 if (m_ave[i] > m_thr_max) m_hot[i] |= 2; // satturated
	 if (m_ave[i] < m_thr_min) m_hot[i] |= 4; // cold
      }
    }
}

//--------------------

double 
NDArrAverage::evaluateThresholdOnRMS()
{
    // calculate mean and sigma usling a few iterations with evolving constraints

    double mean  = 0;
    double width = m_thr_max;
    for (unsigned iter=0; iter<3; ++iter) {
        int    s0 = 0;
        double s1 = 0;
        double s2 = 0;
        for (unsigned i=0; i!=m_size; ++i) {
	        // do not use "cold" and "satturated" pixels:
          	if ( m_ave[i] < m_thr_min || m_ave[i] > m_thr_max ) continue;	

		double dev = m_rms[i] - mean;

                // do not use outlayers:
		if (abs(dev) > width) continue; 
          	s0 += 1;
          	s1 += dev;
          	s2 += dev*dev;
        }
        double ave = (s0>0) ? s1/s0 : 0;
        double rms = (s0>0) ? std::sqrt(s2/s0 - ave*ave) : 0;

	mean = mean + ave;
	width = 5*rms;	

        if( m_print_bits & 128 )	  
            MsgLog(name(), info, "Iteration: " << iter << "  mean: " << mean << "  rms: " << rms);
    }

    double threshold = mean + width;
    if( m_print_bits & 128 )
      MsgLog(name(), info, "Use threshold: " << threshold << " ADU for " << m_str_src);

    return threshold;
}

//--------------------

void 
NDArrAverage::printEventRecord(Event& evt)
{
  MsgLog( name(), info,  "Run="    << m_str_run_num 
                     << " Evt="    << stringFromUint(m_count_ev) 
                     << " Img="    << stringFromUint(m_count) 
                     << " Time="   << stringTimeStamp(evt) 
  );
}

//--------------------

void 
NDArrAverage::printSummaryForParser(Event& evt)
{
  cout << "NDArrAverage: Summary for parser"      << endl;   
  cout << "BATCH_NUMBER_OF_EVENTS " << m_count_ev << endl;
  cout << "BATCH_NUMBER_OF_IMAGES " << m_count    << endl;
  cout << "BATCH_IMG_SIZE         " << m_size     << endl;

  if ( m_ndarr_pars == 0 ) return;
  if ( m_ndarr_pars->ndim() != 2 ) return;

  cout << "BATCH_IMG_ROWS         " << m_ndarr_pars->shape()[0] << endl;
  cout << "BATCH_IMG_COLS         " << m_ndarr_pars->shape()[1] << endl;
}

//--------------------

void 
NDArrAverage::printStatBadPix()
{
  if (m_do_msk || m_do_hot)
    cout << "NUMBER_OF_PIXELS_TOTAL " << m_size    
    	 << "\nNUMBER_OF_PIXELS_BAD   " << m_nbadpix 
    	 << "\nFRACTION_OF_BAD_PIXELS " << fixed << std::setw(8) << std::setprecision(6) << double(m_nbadpix)/m_size
    	 << "\n\n";
}

//--------------------
//--------------------

} // namespace ImgAlgos
