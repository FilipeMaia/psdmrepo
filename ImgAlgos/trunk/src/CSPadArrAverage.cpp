//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadArrAverage...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/CSPadArrAverage.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <fstream>
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
// to work with detector data include corresponding 
// header from psddl_psana package
//#include "psddl_psana/acqiris.ddl.h"

#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace Psana;
using namespace ImgAlgos;

PSANA_MODULE_FACTORY(CSPadArrAverage)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
CSPadArrAverage::CSPadArrAverage (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key()
  , m_aveFile()
  , m_rmsFile()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_str_src = configStr("source",  "DetInfo(:Cspad)");
  m_key     = configStr("key",     "");                 //"calibrated"
  m_aveFile = configStr("avefile", "cspad-ave.dat");
  m_rmsFile = configStr("rmsfile", "cspad-rms.dat");
  m_print_bits = config("print_bits",      0);
  //m_filter  = config   ("filter", false);

  
  // initialize arrays
  std::fill_n(&m_segMask[0], int(MaxQuads), 0U);
  std::fill_n(&m_stat[0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 0);
  std::fill_n(&m_sum [0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 0.);
  std::fill_n(&m_sum2[0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 0.);
}

//--------------
// Destructor --
//--------------
CSPadArrAverage::~CSPadArrAverage ()
{
}

/// Method which is called once at the beginning of the job
void 
CSPadArrAverage::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1<<0 ) printInputParameters();
}

/// Method which is called at the beginning of the run
void 
CSPadArrAverage::beginRun(Event& evt, Env& env)
{
  // Find all configuration objects matching the source address
  // provided in configuration. If there is more than one configuration 
  // object is found then complain and stop.
  
  std::string src = configStr("source", "DetInfo(:Cspad)");
  int count = 0;
  
  // need to know segment mask which is availabale in configuration only
  shared_ptr<Psana::CsPad::ConfigV1> config1 = env.configStore().get(m_str_src, &m_src);
  if (config1.get()) {
    for (int i = 0; i < MaxQuads; ++i) { m_segMask[i] = config1->asicMask()==1 ? 0x3 : 0xff; }
    ++ count;
  }
  
  shared_ptr<Psana::CsPad::ConfigV2> config2 = env.configStore().get(m_str_src, &m_src);
  if (config2.get()) {
    for (int i = 0; i < MaxQuads; ++i) { m_segMask[i] = config2->roiMask(i); }
    ++ count;
  }

  shared_ptr<Psana::CsPad::ConfigV3> config3 = env.configStore().get(m_str_src, &m_src);
  if (config3.get()) {
    for (int i = 0; i < MaxQuads; ++i) { m_segMask[i] = config3->roiMask(i); }
    ++ count;
  }

  if (not count) {
    MsgLog(name(), error, "No CSPad configuration objects found. Terminating.");
    terminate();
    return;
  }
  
  if (count > 1) {
    MsgLog(name(), error, "Multiple CSPad configuration objects found, use more specific source address. Terminating.");
    terminate();
    return;
  }

  MsgLog(name(), info, "Found CSPad object with address " << m_src);
  if (m_src.level() != Pds::Level::Source) {
    MsgLog(name(), error, "Found CSPad configuration object with address not at Source level. Terminating.");
    terminate();
    return;
  }

  const Pds::DetInfo& dinfo = static_cast<const Pds::DetInfo&>(m_src);
  // validate that this is indeed CSPad, should always be true, but
  // additional protection here should not hurt
  if (dinfo.device() != Pds::DetInfo::Cspad) {
    MsgLog(name(), error, "Found CSPad configuration object with invalid address. Terminating.");
    terminate();
    return;
  }
}



/// Method which is called at the beginning of the calibration cycle
void 
CSPadArrAverage::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPadArrAverage::event(Event& evt, Env& env)
{
  shared_ptr<Psana::CsPad::DataV1> data1 = evt.get(m_str_src, m_key, &m_src);
  if (data1.get()) {

    ++ m_count;
    
    int nQuads = data1->quads_shape()[0];
    for (int iq = 0; iq != nQuads; ++ iq) {

      const CsPad::ElementV1& quad = data1->quads(iq);
      const ndarray<int16_t, 3>& data = quad.data();
      collectStat(quad.quad(), data.data());
    }    
  }
  
  shared_ptr<Psana::CsPad::DataV2> data2 = evt.get(m_str_src, m_key, &m_src);
  if (data2.get()) {

    ++ m_count;
    
    int nQuads = data2->quads_shape()[0];
    for (int iq = 0; iq != nQuads; ++ iq) {
      
      const CsPad::ElementV2& quad = data2->quads(iq);
      const ndarray<int16_t, 3>& data = quad.data();
      collectStat(quad.quad(), data.data());
    } 
  }

  if( m_print_bits & 1<<2 ) printEventId(evt);
}
  
/// Method which is called at the end of the calibration cycle
void 
CSPadArrAverage::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
CSPadArrAverage::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
CSPadArrAverage::endJob(Event& evt, Env& env)
{
  MsgLog(name(), info, "collected total " << m_count << " events");
  
  if (not m_aveFile.empty()) {
    // save averaged values in file
    std::ofstream out(m_aveFile.c_str());
    for (int iq = 0; iq != MaxQuads; ++ iq) {
      for (int is = 0; is != MaxSectors; ++ is) {
        for (int ic = 0; ic != NumColumns; ++ ic) {
          for (int ir = 0; ir != NumRows; ++ ir) {
            double avg = m_count ? m_sum[iq][is][ic][ir] / m_count : 0;
            out << avg << ' ';
          }
          out << '\n';
        }
      }
    }
    out.close();
  }

  if (not m_rmsFile.empty()) {    
    // save rms values in file
    std::ofstream out(m_rmsFile.c_str());
    for (int iq = 0; iq != MaxQuads; ++ iq) {
      for (int is = 0; is != MaxSectors; ++ is) {
        for (int ic = 0; ic != NumColumns; ++ ic) {
          for (int ir = 0; ir != NumRows; ++ ir) {
            double stdev = 0;
            if (m_count > 1) {
              double avg = m_sum[iq][is][ic][ir] / m_count;
              stdev = std::sqrt(m_sum2[iq][is][ic][ir] / m_count - avg*avg);
            }
            out << stdev << ' ';
          }
          out << '\n';
        }
      }
    }
    out.close();
  }
}

/// Collect statistics
void 
CSPadArrAverage::collectStat(unsigned quad, const int16_t* data)
{
  //cout << "collectStat for quad =" << quad << endl;

  int ind_in_arr = 0;
  for (int sect = 0; sect < MaxSectors; ++ sect) {
    if (m_segMask[quad] & (1 << sect)) {
     
      // beginning of the segment data
      double* sum  = &m_sum [quad][sect][0][0];
      double* sum2 = &m_sum2[quad][sect][0][0];
      const int16_t* segData = data + ind_in_arr*SectorSize;

      // sum
      for (int i = 0; i < SectorSize; ++ i) {
        sum [i] += double(segData[i]);
        sum2[i] += double(segData[i])*double(segData[i]);
      }          
      
      ++ind_in_arr;
    }
  }
}

//--------------------

// Print input parameters
void 
CSPadArrAverage::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n source     : "     << m_str_src
        << "\n key        : "     << m_key      
        << "\n m_aveFile  : "     << m_aveFile    
        << "\n m_rmsFile  : "     << m_rmsFile    
        << "\n print_bits : "     << m_print_bits
        << "\n";     

    log << "\n MaxQuads  : " << MaxQuads    
        << "\n MaxSectors: " << MaxSectors  
        << "\n NumColumns: " << NumColumns  
        << "\n NumRows   : " << NumRows     
        << "\n SectorSize: " << SectorSize  
        << "\n";
  }
}

//--------------------

void 
CSPadArrAverage::printEventId(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    MsgLog( name(), info, "Event="  << m_count << " ID: " << *eventId);
  }
}

//--------------------

} // namespace ImgAlgos
