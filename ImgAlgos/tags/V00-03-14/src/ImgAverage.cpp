//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgAverage...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/ImgAverage.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <fstream>
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "ImgAlgos/GlobalMethods.h"
//#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace Psana;
using namespace ImgAlgos;

PSANA_MODULE_FACTORY(ImgAverage)

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
ImgAverage::ImgAverage (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key()
  , m_aveFile()
  , m_rmsFile()
  , m_hotFile()
  , m_hot_thr()
  , m_print_bits()
  , m_count(0)
  , m_nev_stage1()
  , m_nev_stage2()
  , m_gate_width1()
  , m_gate_width2()
{
  // get the values from configuration or use defaults
  m_str_src =  configSrc("source",      "DetInfo(:Cspad)");
  m_key     =  configStr("key",         "");
  m_aveFile =  configStr("avefile",     "img-ave.dat");
  m_rmsFile =  configStr("rmsfile",     "img-rms.dat");
  m_hotFile =  configStr("hotpix_mask", "");
  m_hot_thr     = config("hotpix_thr_adu", 10000.);
  m_nev_stage1  = config("evts_stage1",  1000000);
  m_nev_stage2  = config("evts_stage2",        0);
  m_gate_width1 = config("gate_width1",        0); 
  m_gate_width2 = config("gate_width2",        0); 
  m_print_bits  = config("print_bits",         0);
 
  m_do_mask = (m_hotFile.empty()) ? false : true;
 }

//--------------------

// Print input parameters
void 
ImgAverage::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n source     : " << m_str_src
        << "\n key        : " << m_key      
        << "\n m_aveFile  : " << m_aveFile    
        << "\n m_rmsFile  : " << m_rmsFile    
        << "\n m_hotFile  : " << m_hotFile    
        << "\n m_hot_thr  : " << m_hot_thr    
        << "\n m_do_mask  : " << m_do_mask
        << "\n print_bits : " << m_print_bits
        << "\n evts_stage1: " << m_nev_stage1   
        << "\n evts_stage2: " << m_nev_stage2  
        << "\n gate_width1: " << m_gate_width1 
        << "\n gate_width2: " << m_gate_width2 
        << "\n";     
    log << "\n Image shape parameters:"
        << "\n Columns : "    << m_cols  
        << "\n Rows    : "    << m_rows     
        << "\n Size    : "    << m_size  
        << "\n";
  }
}

//--------------
// Destructor --
//--------------
ImgAverage::~ImgAverage ()
{
}

/// Method which is called once at the beginning of the job
void 
ImgAverage::beginJob(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the run
void 
ImgAverage::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
ImgAverage::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ImgAverage::event(Event& evt, Env& env)
{
  ++ m_count;
  if( m_print_bits & 2 ) printEventRecord(evt);
  setCollectionMode(evt);
  collectStat(evt);
}
  
/// Method which is called at the end of the calibration cycle
void 
ImgAverage::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
ImgAverage::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
ImgAverage::endJob(Event& evt, Env& env)
{
  procStatArrays();
  save2DArrayInFile<double> ( m_aveFile, m_ave, m_rows, m_cols, m_print_bits & 16 );
  save2DArrayInFile<double> ( m_rmsFile, m_rms, m_rows, m_cols, m_print_bits & 16 );
  if (m_do_mask) save2DArrayInFile<int> ( m_hotFile, m_hot, m_rows, m_cols, m_print_bits & 16 );
  if( m_print_bits & 16 ) printSummaryForParser(evt);
}

//--------------------

/// Check the event counter and deside what to do next accumulate/change mode/etc.
void 
ImgAverage::setCollectionMode(Event& evt)
{
  // Set the statistics collection mode without gate
  if (m_count == 1 ) {
    defineImageShape(evt, m_str_src, m_key, m_shape); // shape is not available in beginJob and beginRun
    m_rows = m_shape[0];
    m_cols = m_shape[1];
    m_size = m_rows*m_cols;

    if( m_print_bits & 1 ) printInputParameters();
    m_stat = new unsigned[m_size];
    m_sum  = new double  [m_size];
    m_sum2 = new double  [m_size];
    m_ave  = new double  [m_size];
    m_rms  = new double  [m_size]; 
    m_hot  = new int     [m_size]; 
    resetStatArrays();
    m_gate_width = 0;
    if( m_print_bits & 4 ) MsgLog(name(), info, "Stage 0: Event = " << m_count << " Begin to collect statistics without gate.");
  }

  // Change the statistics collection mode for gated stage 1
  else if (m_count == m_nev_stage1 ) {
    procStatArrays();
    resetStatArrays();
    m_gate_width = m_gate_width1;
    if( m_print_bits & 4 ) MsgLog(name(), info, "Stage 1: Event = " << m_count << " Begin to collect statistics with gate =" << m_gate_width);
  } 

  // Change the statistics collection mode for gated stage 2
  else if (m_count == m_nev_stage1 + m_nev_stage2 ) {
    procStatArrays();
    resetStatArrays();
    m_gate_width = m_gate_width2;
    if( m_print_bits & 4 ) MsgLog(name(), info, "Stage 2: Event = " << m_count << " Begin to collect statistics with gate =" << m_gate_width);
  }
}

//--------------------

/// Reset arrays for statistics accumulation
void
ImgAverage::resetStatArrays()
{
  std::fill_n(m_stat, int(m_size), unsigned(0));
  std::fill_n(m_sum,  int(m_size), double(0.));
  std::fill_n(m_sum2, int(m_size), double(0.));
}

//--------------------

/// Collect statistics
void 
ImgAverage::collectStat(Event& evt)
{
  if ( collectStatForType<double>  (evt) ) return;
  if ( collectStatForType<uint16_t>(evt) ) return;
  if ( collectStatForType<int>     (evt) ) return;
  if ( collectStatForType<float>   (evt) ) return;
  if ( collectStatForType<uint8_t> (evt) ) return;
  if ( collectStatForType<short>   (evt) ) return;
  if ( collectStatForType<int16_t> (evt) ) return;

  MsgLog(name(), info, "Image is not available in the event(...) for source:" << m_str_src << " key:" << m_key);
}

//--------------------

/// Process accumulated stat arrays and evaluate m_ave(rage) and m_rms arrays
void 
ImgAverage::procStatArrays()
{
  if( m_print_bits & 8 ) MsgLog(name(), info, "Process statistics for collected total " << m_count << " events");
  
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

    if (m_do_mask) {
      for (unsigned i=0; i!=m_size; ++i)
         m_hot[i] = (m_rms[i] > m_hot_thr) ? 0 : 1;
    }
}

//--------------------

void 
ImgAverage::printEventRecord(Event& evt)
{
  MsgLog( name(), info,  "Run="    << stringRunNumber(evt) 
                     << " Evt="    << stringFromUint(m_count) 
                     << " Time="   << stringTimeStamp(evt) 
  );
}

//--------------------

void 
ImgAverage::printSummaryForParser(Event& evt)
{
  cout << "ImgAverage: Summary for parser" << endl;
  cout << "BATCH_NUMBER_OF_EVENTS " << m_count << endl;
  cout << "BATCH_IMG_ROWS         " << m_rows  << endl;
  cout << "BATCH_IMG_COLS         " << m_cols  << endl;
  cout << "BATCH_IMG_SIZE         " << m_size  << endl;
}

//--------------------

} // namespace ImgAlgos
