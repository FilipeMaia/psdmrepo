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
#include <sstream> // for stringstream

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

using namespace std;

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
  m_size_of_list = 5;
  m_src_list     = new Source[m_size_of_list];
  m_src_list[0]  = configSrc("imon1",       "BldInfo(FEEGasDetEnergy)");
  m_src_list[1]  = configSrc("imon2",       "BldInfo(XCS-IPM-02)");
  m_src_list[2]  = configSrc("imon3",       "BldInfo(XCS-IPM-mono)");
  m_src_list[3]  = configSrc("imon4",       "DetInfo(XcsBeamline.1:Ipimb.4)");
  m_src_list[4]  = configSrc("imon5",       "DetInfo(XcsBeamline.1:Ipimb.5)");

  m_file_type    = configStr("file_type",   "txt");
  m_fname        = configStr("file_data",   "intensity-monitor-data.txt");
  m_fname_header = configStr("file_header", "intensity-monitor-comments.txt");
  m_print_bits   = config   ("print_bits",  0);

  setFileMode();
}

//--------------------

void 
IntensityMonitorsData::setFileMode()
{
  m_file_mode = TEXT;
  if (m_file_type == "bin") m_file_mode = BINARY;
  if (m_file_type == "txt") m_file_mode = TEXT;
}

//--------------------

/// Print input parameters
void 
IntensityMonitorsData::printInputParameters()
{
  WithMsgLog(name(), info, log) {
      log << "\nInput parameters:"
          << "\nSources         : "; 

    for(int i=0; i<m_size_of_list; i++)
      log << "\n  imon" << i+1 << "         : " << m_src_list[i];

      log << "\nfile_type       : " << m_file_type
          << "\nfile_mode       : " << m_file_mode
          << "\nfname           : " << m_fname
          << "\nfname_header    : " << m_fname_header
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
  closeOutputFiles();
  if( m_print_bits & 4 ) printSummary(evt, "");
  if( m_print_bits & 8 ) printSummaryForParser(evt);
}

//--------------------
//--------------------
//--------------------
//--------------------

std::string 
IntensityMonitorsData::strOfSources()
{
  std::stringstream ss;
  for(int i=0; i<m_size_of_list; i++)
    ss << m_src_list[i] << " ";
  ss << "\n";
  return ss.str();
}

//--------------------

/// Open temporary output file with time records
void 
IntensityMonitorsData::openOutputFiles()
{
  if      (m_file_mode == TEXT  ) p_out.open(m_fname.c_str());
  else if (m_file_mode == BINARY) p_out.open(m_fname.c_str(), ios_base::out | ios_base::binary);

  p_out_header.open(m_fname_header.c_str());
  if( m_print_bits & 32 ) MsgLog( name(), info, "Open file: " << m_fname_header.c_str());  
  if( m_print_bits & 32 ) MsgLog( name(), info, "Open file: " << m_fname       .c_str());  

  p_out_header << "Heder for the data file: " << m_fname       .c_str()
               << "\nNumber of sources: " << m_size_of_list
               << "\nFour values per source:\n" 
               << strOfSources();
}

//--------------------

/// Close temporary output file with time records
void 
IntensityMonitorsData::closeOutputFiles()
{
  p_out_header << "Number of records in file: " << m_count;
  p_out_header.close();
  p_out.close();
  if( m_print_bits & 32 ) MsgLog( name(), info, "Monitors data is saved in the file: " << m_fname.c_str());  
  if( m_print_bits & 32 ) MsgLog( name(), info, "Headier       is saved in the file: " << m_fname_header.c_str());  
}

//--------------------

void 
IntensityMonitorsData::procEvent(Event& evt, Env& env)
{  

  if( m_print_bits & 16 ) // Print all available data for list of sources
    for(int i=0; i<m_size_of_list; i++) {
      printDataForSource(evt, env, m_src_list[i]);
    }

  if (m_file_mode == TEXT) {
    std::string s = strRecord(evt, env);
    p_out.write(s.c_str(), s.size());
    //cout  << s.size() << " " << s;
  }

  else if (m_file_mode == BINARY) {
    float* a = arrRecord(evt, env);
    // for (int i=0; i<m_size_of_arr; i++) cout << " " << a[i]; cout << endl;
    p_out.write(reinterpret_cast<const char*>(a), m_size_of_arr*sizeof(float));
    p_out << "\n";
  } 
}

//--------------------

std::string 
IntensityMonitorsData::strRecord(Event& evt, Env& env)
{
  std::stringstream ss;
  ss << right << std::setw(7) << m_count-1 << "  " << fixed << std::setprecision(5); 

  for(int i=0; i<m_size_of_list; i++) {

      Quartet q = getDataForSource(evt, env, m_src_list[i]);
      //cout << m_src_list[i] << ": " << q.v1 << " " << q.v2 << " " << q.v3 << " " << q.v4 << endl; 
      ss << q.v1 << " " 
         << q.v2 << " " 
         << q.v3 << " " 
         << q.v4 << "  ";
  }
  ss << "\n";
  return ss.str();
}

//--------------------

float*
IntensityMonitorsData::arrRecord(Event& evt, Env& env)
{
  m_size_of_arr = 4*m_size_of_list + 1;
  float* arr = new float[m_size_of_arr];
  arr[0] = (float)m_count-1;

  for(int i=0, ibase=0; i<m_size_of_list; i++, ibase+=4) {

      Quartet q = getDataForSource(evt, env, m_src_list[i]);
      arr[ibase+1] = q.v1;
      arr[ibase+2] = q.v2;
      arr[ibase+3] = q.v3;
      arr[ibase+4] = q.v4;
  }
  return arr;
}

//-------------------- 

Quartet
IntensityMonitorsData::getDataForSource(Event& evt, Env& env, Source& src)
{  
  shared_ptr<Psana::Bld::BldDataFEEGasDetEnergy> fee = evt.get(src);
  if (fee.get()) {

    if( m_print_bits & 64 ) MsgLog( name(), info, "get: " << src << " from Bld::BldDataFEEGasDetEnergy, f_ij_ENRC(): " 
				    << " " << fee->f_11_ENRC()
				    << " " << fee->f_12_ENRC()
				    << " " << fee->f_21_ENRC()
				    << " " << fee->f_22_ENRC()
    );

    return Quartet ((float)fee->f_11_ENRC(), 
                    (float)fee->f_12_ENRC(), 
                    (float)fee->f_21_ENRC(), 
                    (float)fee->f_22_ENRC());
  } 

//----- for XCS-IPM-02, XCS-IPM-mono, XcsBeamline.1:Ipimb.4, XcsBeamline.1:Ipimb.5, see: psana_examples/src/DumpIpimb.cpp

  shared_ptr<Psana::Lusi::IpmFexV1> fex = evt.get(src);
  if (fex) {

    if( m_print_bits & 64 ) MsgLog( name(), info, "get: " << src << " from Lusi::IpmFexV1" << " channel =" << fex->channel() );

    return Quartet ((float)fex->channel()[0], 
                    (float)fex->channel()[1], 
                    (float)fex->channel()[2], 
                    (float)fex->channel()[3]);
  }

 
  shared_ptr<Psana::Ipimb::DataV2> data2 = evt.get(src);
  if (data2.get()) {

    if( m_print_bits & 64 ) MsgLog( name(), info, "get: " << src << " from Ipimb::DataV2" );

    return Quartet ((float)data2->channel0Volts(), 
                    (float)data2->channel1Volts(), 
                    (float)data2->channel2Volts(), 
                    (float)data2->channel3Volts());
  }

//----- for XCS-IPM-02 and XCS-IPM-mono, see psana_examples/src/DumpBld.cpp

  shared_ptr<Psana::Bld::BldDataIpimbV1> ipimb1 = evt.get(src); 
  if (ipimb1.get()) {

    const Psana::Lusi::IpmFexV1& ipmFexData = ipimb1->ipmFexData();

    if( m_print_bits & 64 ) MsgLog( name(), info, "get: " << src << " from BldDataIpimbV1" << ipmFexData.channel() );

    return Quartet ((float)ipmFexData.channel()[0], 
                    (float)ipmFexData.channel()[1], 
                    (float)ipmFexData.channel()[2], 
                    (float)ipmFexData.channel()[3]);
    /*
    const Psana::Ipimb::DataV2& ipimbData = ipimb1->ipimbData();
    return Quartet ((float)ipimbData.channel0Volts(), 
                    (float)ipimbData.channel1Volts(), 
                    (float)ipimbData.channel2Volts(), 
                    (float)ipimbData.channel3Volts());
    */
  }

//-----
  MsgLog( name(), info,  "IntensityMonitorsData::getDataForSource(): unavailable data for source: " << src << "\n"); 
  //abort(); 
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
//--------------------

} // namespace ImgAlgos
