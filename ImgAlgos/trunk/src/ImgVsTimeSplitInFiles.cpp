//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgVsTimeSplitInFiles...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/ImgVsTimeSplitInFiles.h"

//-----------------
// C/C++ Headers --
//-----------------
// #include <time.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEvt/EventId.h"
#include "ImgAlgos/GlobalMethods.h"
#include "ImgAlgos/TimeInterval.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
//#include <boost/lexical_cast.hpp>
#include <iomanip> // for setw, setfill
#include <sstream> // for stringstream
#include <cmath> // for sqrt, atan2, etc.
#include <boost/lexical_cast.hpp>
//#include <typeinfo> // for typeid(m_data).name()

// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(ImgVsTimeSplitInFiles)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------

ImgVsTimeSplitInFiles::ImgVsTimeSplitInFiles (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key()
  , m_fname_prefix()
  , m_file_type()
  , m_add_tstamp()
  , m_nfiles_out()
  , m_ampl_thr()
  , m_ampl_min()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_str_src       = configSrc("source", "DetInfo(:Princeton)");
  m_key           = configStr("key",                 "img");
  m_fname_prefix  = configStr("fname_prefix",     "my-exp");
  m_file_type     = configStr("file_type",           "bin");
  m_add_tstamp    = config   ("add_tstamp",           true);
  m_nfiles_out    = config   ("nfiles_out",              8);
  m_ampl_thr      = config   ("ampl_thr",                1);
  m_ampl_min      = config   ("ampl_min",                1);
  m_print_bits    = config   ("print_bits",              0);

  setFileMode();
}

//--------------------
/// Print input parameters
void 
ImgVsTimeSplitInFiles::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\nInput parameters:"
        << "\nsource       : " << m_str_src
        << "\nkey          : " << m_key      
        << "\nfname_prefix : " << m_fname_prefix
        << "\nfile_type    : " << m_file_type
        << "\nm_file_mode  : " << m_file_mode
        << "\nadd_tstamp   : " << m_add_tstamp
        << "\nnfiles_out   : " << m_nfiles_out
        << "\nm_ampl_thr   : " << m_ampl_thr
        << "\nm_ampl_min   : " << m_ampl_min
        << "\nm_print_bits : " << m_print_bits;
  }
}

//--------------------

void 
ImgVsTimeSplitInFiles::setFileMode()
{
  m_file_mode = TEXT;
  if (m_file_type == "bin") m_file_mode = BINARY;
  if (m_file_type == "txt") m_file_mode = TEXT;
}

///--------------------


//--------------
// Destructor --
//--------------

ImgVsTimeSplitInFiles::~ImgVsTimeSplitInFiles ()
{
}

//--------------------

//--------------------

/// Method which is called once at the beginning of the job
void 
ImgVsTimeSplitInFiles::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
}

//--------------------

/// Method which is called at the beginning of the run
void 
ImgVsTimeSplitInFiles::beginRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the beginning of the calibration cycle
void 
ImgVsTimeSplitInFiles::beginCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ImgVsTimeSplitInFiles::event(Event& evt, Env& env)
{
  m_count++;
  if(m_count==1){
    initSplitInFiles(evt, env);
    openOutputFiles(evt);
  }

  procEvent(evt);
  if( m_print_bits & 2 ) printEventRecord(evt, " is split and saved");
}

//--------------------
  
/// Method which is called at the end of the calibration cycle
void 
ImgVsTimeSplitInFiles::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the end of the run
void 
ImgVsTimeSplitInFiles::endRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called once at the end of the job
void 
ImgVsTimeSplitInFiles::endJob(Event& evt, Env& env)
{
  closeOutputFiles();
  evaluateMeanTimeBetweenEvents();
  saveTimeRecordWithIndexInFile();
  saveMetadataInFile();
  if( m_print_bits & 4 ) printSummary(evt, " images are splitted");
}

//--------------------
//--------------------
//--------------------

void 
ImgVsTimeSplitInFiles::initSplitInFiles(Event& evt, Env& env)
{
    if( initSplitInFilesForType<double>  (evt, env) ) return;
    if( initSplitInFilesForType<float>   (evt, env) ) return;
    if( initSplitInFilesForType<int>     (evt, env) ) return;
    if( initSplitInFilesForType<uint16_t>(evt, env) ) return;

    const std::string msg = "Image shape is not defined in the event(...) for source:" 
                          + boost::lexical_cast<std::string>(m_str_src) + " key:" + m_key;
    MsgLogRoot(error, msg);
    throw std::runtime_error(msg);
}

//--------------------

void 
ImgVsTimeSplitInFiles::saveMetadataInFile()
{
  std::string fname = m_fname_common+"-med.txt";
  std::ofstream out(fname.c_str());

  out <<   "IMAGE_ROWS      " << m_img_rows 
      << "\nIMAGE_COLS      " << m_img_cols
      << "\nIMAGE_SIZE      " << m_img_size
      << "\nNUMBER_OF_FILES " << m_nfiles_out
      << "\nBLOCK_SIZE      " << m_blk_size
      << "\nREST_SIZE       " << m_rst_size
      << "\nNUMBER_OF_IMGS  " << m_count
      << "\nFILE_TYPE       " << m_file_type
    //<< "\nDATA_TYPE_INPUT " << m_data_type_input
      << "\nDATA_TYPE       " << typeid(data_split_t).name() // i.e "f" or "d"
      << "\nTIME_SEC_AVE    " << fixed << std::setprecision(6) << m_t_ave
      << "\nTIME_SEC_RMS    " << fixed << std::setprecision(6) << m_t_rms
      << "\nTIME_INDEX_MAX  " << std::setw(8) << m_tind_max
      << "\n";

  out.close();

  if( m_print_bits & 16 ) MsgLog( name(), info, "The file with metadata: " << fname << " is created.");
}

//--------------------

void 
ImgVsTimeSplitInFiles::openOutputFiles(Event& evt)
{
  p_out = new std::ofstream [m_nfiles_out];

  ios_base::openmode mode = ios_base::out | ios_base::binary;
  TimeInterval* time_job_start = new TimeInterval();

  m_fname_common = m_fname_prefix + "-r"  + stringRunNumber(evt);
  if(m_add_tstamp)  m_fname_common += time_job_start -> strStartTime("-%Y%m%d-%H%M%S"); 

  std::string fname;
  for(unsigned b=0; b<m_nfiles_out; b++){

     fname = m_fname_common
           + "-b"  + stringFromUint(b,4)
           + "."   + m_file_type;

     if( m_print_bits & 16 ) MsgLog( name(), info, "Open output file: " << fname );     

     p_out[b].open(fname.c_str(), mode);
     //p_out[b] << "This is a content of the file " << fname;
     //p_out[b].close();
  }

  m_fname_time = m_fname_common + "-time.txt";
  p_out_time.open(m_fname_time.c_str());
}

//--------------------

void 
ImgVsTimeSplitInFiles::closeOutputFiles()
{
  for(unsigned b=0; b<m_nfiles_out; b++) p_out[b].close();
  p_out_time.close();
}

//--------------------

void 
ImgVsTimeSplitInFiles::procEvent(Event& evt)
{
  saveTimeRecord(evt);

  if ( procEventForType<double>   (evt) ) return;
  if ( procEventForType<float>    (evt) ) return;
  if ( procEventForType<int>      (evt) ) return;
  if ( procEventForType<uint16_t> (evt) ) return;

  MsgLog(name(), info, "Image is not available in the event(...) for source:" << m_str_src << " key:" << m_key);
}

//--------------------

void 
ImgVsTimeSplitInFiles::evaluateMeanTimeBetweenEvents()
{
  m_t_ave = (m_sumt0) ? m_sumt1/m_sumt0 : 0;
  m_t_rms = (m_sumt0) ? std::sqrt(m_sumt2/m_sumt0 - m_t_ave*m_t_ave) : 0;
}

//--------------------

void 
ImgVsTimeSplitInFiles::saveTimeRecord(Event& evt)
{
  m_tsec = doubleTime(evt);
  m_nevt = eventCounterSinceConfigure(evt);

  if(m_count==1) {
    m_tsec_0    = m_tsec;
    m_tsec_prev = m_tsec;
    m_nevt_prev = m_nevt;
    m_sumt0 = 0;
    m_sumt1 = 0;
    m_sumt2 = 0;
  }

  m_dt = m_tsec-m_tsec_prev;
  
  if ( (m_nevt-m_nevt_prev)==1 ) {
    m_sumt0 ++;
    m_sumt1 += m_dt;
    m_sumt2 += m_dt*m_dt;
  }

  p_out_time << std::setw(6) << m_count-1 // Save the event index starting from 0. 
             << fixed << std::setw(16) << std::setprecision(6) << m_tsec - m_tsec_0
             << fixed << std::setw(10) << std::setprecision(6) << m_dt
             << stringTimeStamp(evt,"  %Y%m%d-%H%M%S%f")
             << std::setw(8) << fiducials(evt)
             << std::setw(7) << m_nevt
             << "\n";

  m_tsec_prev = m_tsec;  
  m_nevt_prev = m_nevt;
}

//--------------------

void 
ImgVsTimeSplitInFiles::printEventRecord(Event& evt, std::string comment)
{
  MsgLog( name(), info,  "Run="    << stringRunNumber(evt) 
                     << " Evt="    << stringFromUint(m_count) 
                     << " Time="   << stringTimeStamp(evt) 
	             << comment.c_str() 
  );
}

//--------------------

void 
ImgVsTimeSplitInFiles::printSummary(Event& evt, std::string comment)
{
  MsgLog( name(), info, "Run=" << stringRunNumber(evt) 
	                << " Number of processed events=" << stringFromUint(m_count)
                        << comment.c_str()
  );
}

//--------------------
// 1. read the time record from the file m_fname_time 
// 2. from t_sec and m_t_ave evaluate the time index tind
// 3. save the time record with time index in another file m_fname_time_ind.
// - In CorAnaData this file is used to associate the time index with event index
void  
ImgVsTimeSplitInFiles::saveTimeRecordWithIndexInFile()
{
  m_fname_time_ind = m_fname_common + "-time-ind.txt";

  MsgLog( name(), info,  "CorAna::readTimeRecordsFile(): Read time records from the file      : " << m_fname_time 
	              << "\n                             and save them with time-index in file: " << m_fname_time_ind );
 
  std::string s;
  unsigned    evind;
  double      t_sec;
  double      dt_sec;
  std::string tstamp;
  unsigned    fiduc;
  unsigned    evnum;
  unsigned    tind;
  double      rind;

  m_tind_max = 0;

  std::ifstream fin(m_fname_time.c_str());
  if (! fin.is_open()) {
    const std::string msg = "Unable to open input file: " + m_fname_time;
    MsgLogRoot(error, msg);
    throw std::runtime_error(msg);
  }

  std::ofstream fout(m_fname_time_ind.c_str());
  if (! fout.is_open()) {
    const std::string msg = "Unable to open output file: " + m_fname_time_ind;
    MsgLogRoot(error, msg);
    throw std::runtime_error(msg);
  }

  while ( true )
  {
    getline (fin,s);
    if(!fin.good()) break;
    std::stringstream ss(s); 
    ss >> evind >> t_sec >> dt_sec >> tstamp >> fiduc >> evnum;

    // === Time index of the event evaluation ===
    rind = (t_sec + 0.5*m_t_ave) / m_t_ave;
    tind = (rind>0) ? static_cast<unsigned>(rind) : 0;
    if (tind > m_tind_max) m_tind_max = tind;
    
    fout << s << fixed << std::setw(9) << tind << " \n";
  }

  fin .close();
  fout.close();
}

//--------------------
//--------------------
//--------------------
//--------------------

} // namespace ImgAlgos
