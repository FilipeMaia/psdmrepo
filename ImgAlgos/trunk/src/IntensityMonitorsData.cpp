//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class IntensityMonitorsData...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/IntensityMonitorsData.h"

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
#include "psddl_psana/bld.ddl.h"
//#include "ImgAlgos/TimeInterval.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
//#include <boost/lexical_cast.hpp>
#include <iomanip> // for setw, setfill
#include <sstream> // for stringstream
#include <cmath> // for sqrt, atan2, etc.

// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(IntensityMonitorsData)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------

IntensityMonitorsData::IntensityMonitorsData (const std::string& name)
  : Module(name)
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_srcFEEGasDetE = configStr("feeSource", "BldInfo(FEEGasDetEnergy)");
  m_srcIPM2       = configStr("ipm2",      "BldInfo(XCS-IPM-02)");
  m_srcIPMMono    = configStr("ipmmono",   "BldInfo(XCS-IPM-mono)");
  m_srcIPM4       = configStr("ipm4",      "DetInfo(XcsBeamline.1:Ipimb.4)");
  m_srcIPM5       = configStr("ipm5",      "DetInfo(XcsBeamline.1:Ipimb.5)");
  m_fname         = configStr("out_file",  "intens-mon-data.txt");
  m_print_bits    = config   ("print_bits",                    0);
}

//--------------------

/// Print input parameters
void 
IntensityMonitorsData::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\nInput parameters:"
        << "\nSources         : " 
        << "\nFEEGasDetE      : " << m_srcFEEGasDetE
        << "\nIPM2            : " << m_srcIPM2      
        << "\nIPMMono         : " << m_srcIPMMono   
        << "\nIPM4            : " << m_srcIPM4      
        << "\nIPM5            : " << m_srcIPM5      
        << "\nfname_prefix    : " << m_fname
        << "\nm_print_bits    : " << m_print_bits;
  }
}

//--------------------

/// Destructor
IntensityMonitorsData::~IntensityMonitorsData ()
{
}

//--------------------

/// Method which is called once at the beginning of the job
void 
IntensityMonitorsData::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
}

//--------------------

/// Method which is called at the beginning of the run
void 
IntensityMonitorsData::beginRun(Event& evt, Env& env)
{
  m_str_run_number  = stringRunNumber(evt);
}

//--------------------

/// Method which is called at the beginning of the calibration cycle
void 
IntensityMonitorsData::beginCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
IntensityMonitorsData::event(Event& evt, Env& env)
{
  m_count++;
  if(m_count==1){
    openOutputFiles();
  }

  procEvent(evt, env);

  if( m_print_bits & 2 ) printEventRecord(evt, "");
}

//--------------------
  
/// Method which is called at the end of the calibration cycle
void 
IntensityMonitorsData::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the end of the run
void 
IntensityMonitorsData::endRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called once at the end of the job
void 
IntensityMonitorsData::endJob(Event& evt, Env& env)
{
  //closeOutputFiles();
  if( m_print_bits & 4 ) printSummary(evt, "");
  if( m_print_bits & 8 ) printSummaryForParser(evt);
}

//--------------------
//--------------------
//--------------------
//--------------------

void 
IntensityMonitorsData::procEvent(Event& evt, Env& env)
{  
  Source src_list[] = {m_srcFEEGasDetE, m_srcIPM2, m_srcIPMMono, m_srcIPM4, m_srcIPM5};

  std::string rec = "";

  for(int i=0; i<5; i++) {
    if( m_print_bits & 16 ) printDataForSource(evt, env, src_list[i]);

    Quartet q = getDataForSource(evt, env, src_list[i]);

    //cout << src_list[i] << ": " << q.v1 << " " << q.v2 << " " << q.v3 << " " << q.v4 << endl; 

    p_out << q.v1 << " " << q.v2 << " " << q.v3 << " " << q.v4 << " ";
  }

  p_out << "\n";
}

//--------------------

Quartet
IntensityMonitorsData::getDataForSource(Event& evt, Env& env, Source& src)
{  
  shared_ptr<Psana::Bld::BldDataFEEGasDetEnergy> fee = evt.get(src);
  if (fee.get()) {
    return Quartet ((float)fee->f_11_ENRC(), 
                    (float)fee->f_12_ENRC(), 
                    (float)fee->f_21_ENRC(), 
                    (float)fee->f_22_ENRC());
  } 

//-----
 
  shared_ptr<Psana::Ipimb::DataV2> data2 = evt.get(src);
  if (data2.get()) {
    return Quartet ((float)data2->channel0Volts(), 
                    (float)data2->channel1Volts(), 
                    (float)data2->channel2Volts(), 
                    (float)data2->channel3Volts());
  }

//-----

  shared_ptr<Psana::Bld::BldDataIpimbV1> ipimb1 = evt.get(src);
  if (ipimb1.get()) {
    const Psana::Ipimb::DataV2& ipimbData = ipimb1->ipimbData();

    return Quartet ((float)ipimbData.channel0Volts(), 
                    (float)ipimbData.channel1Volts(), 
                    (float)ipimbData.channel2Volts(), 
                    (float)ipimbData.channel3Volts());
  }

//-----
    return  Quartet (-1,-1,-1,-1);
}

//--------------------

void 
IntensityMonitorsData::printDataForSource(Event& evt, Env& env, Source& src)
{  
  shared_ptr<Psana::Bld::BldDataFEEGasDetEnergy> fee = evt.get(src);
  if (fee.get()) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataFEEGasDetEnergy:  " << src
          << "\n  f_11_ENRC=" << fee->f_11_ENRC()
          << "\n  f_12_ENRC=" << fee->f_12_ENRC()
          << "\n  f_21_ENRC=" << fee->f_21_ENRC()
          << "\n  f_22_ENRC=" << fee->f_22_ENRC();
    }
    return;
  } 

//-----
 
  shared_ptr<Psana::Ipimb::DataV2> data2 = evt.get(src);
  if (data2.get()) {
    
    WithMsgLog(name(), info, str) {
      str << "Ipimb::DataV2:  " << src
          << "\n  triggerCounter = " << data2->triggerCounter()
          << "\n  config = " << data2->config0()
          << "," << data2->config1()
          << "," << data2->config2()
          << "\n  channel = " << data2->channel0()
          << "," << data2->channel1()
          << "," << data2->channel2()
          << "," << data2->channel3()
          << "\n  volts = " << data2->channel0Volts()
          << "," << data2->channel1Volts()
          << "," << data2->channel2Volts()
          << "," << data2->channel3Volts()
          << "\n  channel-ps = " << data2->channel0ps()
          << "," << data2->channel1ps()
          << "," << data2->channel2ps()
          << "," << data2->channel3ps()
          << "\n  volts-ps = " << data2->channel0psVolts()
          << "," << data2->channel1psVolts()
          << "," << data2->channel2psVolts()
          << "," << data2->channel3psVolts()
          << "\n  checksum = " << data2->checksum();
    }
    return;
  }

//-----

  shared_ptr<Psana::Bld::BldDataIpimbV1> ipimb1 = evt.get(src);
  if (ipimb1.get()) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataIpimbV1:  " << src;
      const Psana::Ipimb::DataV2& ipimbData = ipimb1->ipimbData();
      str << "\n  Ipimb::DataV1:"
          << "\n    triggerCounter = " << ipimbData.triggerCounter()
          << "\n    config = " << ipimbData.config0()
          << "," << ipimbData.config1()
          << "," << ipimbData.config2()
          << "\n    channel = " << ipimbData.channel0()
          << "," << ipimbData.channel1()
          << "," << ipimbData.channel2()
          << "," << ipimbData.channel3()
          << "\n    volts = " << ipimbData.channel0Volts()
          << "," << ipimbData.channel1Volts()
          << "," << ipimbData.channel2Volts()
          << "," << ipimbData.channel3Volts()
          << "\n    channel-ps = " << ipimbData.channel0ps()
          << "," << ipimbData.channel1ps()
          << "," << ipimbData.channel2ps()
          << "," << ipimbData.channel3ps()
          << "\n    volts-ps = " << ipimbData.channel0psVolts()
          << "," << ipimbData.channel1psVolts()
          << "," << ipimbData.channel2psVolts()
          << "," << ipimbData.channel3psVolts()
          << "\n    checksum = " << ipimbData.checksum();
    }
    return;
  }

//-----
  MsgLog( name(), info, "Not found data for Psana::Bld::BldDataIpimbV1: " << src);
}

//--------------------

/// Open temporary output file with time records
void 
IntensityMonitorsData::openOutputFiles()
{
  p_out.open(m_fname.c_str());
}

//--------------------

/// Close temporary output file with time records
void 
IntensityMonitorsData::closeOutputFiles()
{
  p_out.close();
}

//--------------------

/// Evaluate average time and rms between the frames
//void 
//IntensityMonitorsData::evaluateMeanTimeBetweenEvents()
//{
//  m_t_ave = (m_sumt0) ? m_sumt1/m_sumt0 : 0;
//  m_t_rms = (m_sumt0) ? std::sqrt(m_sumt2/m_sumt0 - m_t_ave*m_t_ave) : 0;
//}

//--------------------

/// Saves the time record in temporary output file
/*
void 
IntensityMonitorsData::saveTimeRecord(Event& evt)
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

  p_out      << std::setw(6) << m_count-1 // Save the event index starting from 0. 
             << fixed << std::setw(16) << std::setprecision(6) << m_tsec - m_tsec_0
             << fixed << std::setw(10) << std::setprecision(6) << m_dt
             << stringTimeStamp(evt,"  %Y%m%d-%H%M%S%f")
             << std::setw(8) << fiducials(evt)
             << std::setw(7) << m_nevt
             << "\n";

  m_tsec_prev = m_tsec;  
  m_nevt_prev = m_nevt;
}
*/

//--------------------

/// Print event record
void 
IntensityMonitorsData::printEventRecord(Event& evt, std::string comment)
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
IntensityMonitorsData::printSummary(Event& evt, std::string comment)
{
  MsgLog( name(), info, "Run=" << m_str_run_number 
	                << " Number of processed events=" << stringFromUint(m_count)
                        << comment.c_str()
  );
}

//--------------------

/// Print summary for parser
void 
IntensityMonitorsData::printSummaryForParser(Event& evt, std::string comment)
{
  cout << "IntensityMonitorsData: Summary for parser " << comment.c_str() << endl;
  cout << "BATCH_RUN_NUMBER              " << m_str_run_number << endl;
  cout << "BATCH_NUMBER_OF_EVENTS        " << m_count << endl;
}

//--------------------

/// Save metadata in file 
//void  
//IntensityMonitorsData::saveMetadataInFile()
//{
//  std::string fname = m_fname+"-med.txt";
//  std::ofstream out(fname.c_str());

//  out << "\nTIME_SEC_AVE    " << fixed << std::setprecision(6) << m_t_ave
//      << "\nTIME_SEC_RMS    " << fixed << std::setprecision(6) << m_t_rms
//      << "\nTIME_INDEX_MAX  " << std::setw(8) << m_tind_max
//      << "\n";

//  out.close();
//  if( m_print_bits & 16 ) MsgLog( name(), info, "The file with metadata: " << fname << " is created.");
//}

//--------------------
//--------------------
//--------------------
//--------------------

} // namespace ImgAlgos
