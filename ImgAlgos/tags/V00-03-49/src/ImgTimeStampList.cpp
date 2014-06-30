//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgTimeStampList...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/ImgTimeStampList.h"

//-----------------
// C/C++ Headers --
//-----------------
// #include <time.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
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

// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(ImgTimeStampList)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------

ImgTimeStampList::ImgTimeStampList (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key()
  , m_fname()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_str_src       = configSrc("source", "DetInfo(:Princeton)");
  m_key           = configStr("key",                       "");
  m_fname         = configStr("out_file",   "tstamp-list.txt");
  m_print_bits    = config   ("print_bits",                 0);
}

//--------------------

/// Print input parameters
void 
ImgTimeStampList::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\nInput parameters:"
        << "\nsource       : " << m_str_src
        << "\nkey          : " << m_key      
        << "\nfname_prefix : " << m_fname
        << "\nm_print_bits : " << m_print_bits;
  }
}

//--------------------

/// Destructor
ImgTimeStampList::~ImgTimeStampList ()
{
}

//--------------------

/// Method which is called once at the beginning of the job
void 
ImgTimeStampList::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
}

//--------------------

/// Method which is called at the beginning of the run
void 
ImgTimeStampList::beginRun(Event& evt, Env& env)
{
  m_str_run_number  = stringRunNumber(evt);
}

//--------------------

/// Method which is called at the beginning of the calibration cycle
void 
ImgTimeStampList::beginCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ImgTimeStampList::event(Event& evt, Env& env)
{
  m_count++;
  if(m_count==1){
    openOutputFiles(evt);
  }

  saveTimeRecord(evt);

  if( m_print_bits & 2 ) printEventRecord(evt, "");
}

//--------------------
  
/// Method which is called at the end of the calibration cycle
void 
ImgTimeStampList::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the end of the run
void 
ImgTimeStampList::endRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called once at the end of the job
void 
ImgTimeStampList::endJob(Event& evt, Env& env)
{
  closeOutputFiles();
  evaluateMeanTimeBetweenEvents();
  saveTimeRecordWithIndexInFile();
  //saveMetadataInFile();
  if( m_print_bits & 4 ) printSummary(evt, "");
  if( m_print_bits & 8 ) printSummaryForParser(evt);
}

//--------------------
//--------------------
//--------------------
//--------------------

/// Open temporary output file with time records
void 
ImgTimeStampList::openOutputFiles(Event& evt)
{
  m_fname_time = m_fname + "-tmp";
  p_out_time.open(m_fname_time.c_str());
}

//--------------------

/// Close temporary output file with time records
void 
ImgTimeStampList::closeOutputFiles()
{
  p_out_time.close();
}

//--------------------

/// Evaluate average time and rms between the frames
void 
ImgTimeStampList::evaluateMeanTimeBetweenEvents()
{
  m_t_ave = (m_sumt0) ? m_sumt1/m_sumt0 : 0;
  m_t_rms = (m_sumt0) ? std::sqrt(m_sumt2/m_sumt0 - m_t_ave*m_t_ave) : 0;
}

//--------------------

/// Saves the time record in temporary output file
void 
ImgTimeStampList::saveTimeRecord(Event& evt)
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

/// Print event record
void 
ImgTimeStampList::printEventRecord(Event& evt, std::string comment)
{
  MsgLog( name(), info,  "Run="    << m_str_run_number
                     << " Evt="    << stringFromUint(m_count) 
                     << " Time="   << stringTimeStamp(evt) 
	             << comment.c_str() 
  );
}

//--------------------

/// Print summary
void 
ImgTimeStampList::printSummary(Event& evt, std::string comment)
{
  MsgLog( name(), info, "Run=" << m_str_run_number 
	                << " Number of processed events=" << stringFromUint(m_count)
                        << comment.c_str()
  );
}

//--------------------

/// Print summary for parser
void 
ImgTimeStampList::printSummaryForParser(Event& evt, std::string comment)
{
  cout << "ImgTimeStampList: Summary for parser " << comment.c_str() << endl;
  cout << "BATCH_RUN_NUMBER              " << m_str_run_number << endl;
  cout << "BATCH_NUMBER_OF_EVENTS        " << m_count << endl;
  cout << "BATCH_FRAME_TIME_INTERVAL_AVE " << fixed << std::setprecision(6) << m_t_ave << endl;
  cout << "BATCH_FRAME_TIME_INTERVAL_RMS " << fixed << std::setprecision(6) << m_t_rms << endl;
  cout << "BATCH_FRAME_TIME_INDEX_MAX    " << std::setw(8) << m_tind_max << endl;
}

//--------------------

/// Save metadata in file 
void  
ImgTimeStampList::saveMetadataInFile()
{
  std::string fname = m_fname+"-med.txt";
  std::ofstream out(fname.c_str());

  out << "\nTIME_SEC_AVE    " << fixed << std::setprecision(6) << m_t_ave
      << "\nTIME_SEC_RMS    " << fixed << std::setprecision(6) << m_t_rms
      << "\nTIME_INDEX_MAX  " << std::setw(8) << m_tind_max
      << "\n";

  out.close();
  if( m_print_bits & 16 ) MsgLog( name(), info, "The file with metadata: " << fname << " is created.");
}

//--------------------
/// Functionality:
/// 1. read the time record from the file m_fname_time 
/// 2. from t_sec and m_t_ave evaluate the time index tind
/// 3. save the time record with time index in the output file m_fname.
/// - In CorAnaData this file is used to associate the time index with event index
void  
ImgTimeStampList::saveTimeRecordWithIndexInFile()
{
  MsgLog( name(), info,  "CorAna::readTimeRecordsFile(): Read time records from the file      : " << m_fname_time 
	              << "\n                             and save them with time-index in file: " << m_fname );
 
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

  std::ofstream fout(m_fname.c_str());
  if (! fout.is_open()) {
    const std::string msg = "Unable to open output file: " + m_fname;
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
