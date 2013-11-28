//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadPedestals...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "cspad_mod/CsPadPedestals.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <fstream>
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace Psana;

// This declares this class as psana module
using namespace cspad_mod;
PSANA_MODULE_FACTORY(CsPadPedestals)

namespace {
  
  // returns squared standard deviation calculated from sum and sum of squares
  double stddev2(unsigned count, int64_t sum, int64_t sum2) 
  {
    // if we just convert numbers to doubles then precision may not be 
    // enough in case of large mean values, so we offset all values first 
    // to bring them closer to mean values
    int64_t offset = sum / count;
    int64_t sum2o = sum2 - 2*offset*sum + count*offset*offset;
    int64_t sumo = sum - offset*count;
    double avg = double(sumo) / count;
    return double(sum2o) / count - avg*avg;
  }
  
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace cspad_mod {

//----------------
// Constructors --
//----------------
CsPadPedestals::CsPadPedestals (const std::string& name)
  : Module(name)
  , m_pedFile()
  , m_noiseFile()
  , m_src()
  , m_segMask()
  , m_sum()
  , m_sum2()
{
  m_pedFile = configStr("output", "cspad-pedestals.dat");
  m_noiseFile = configStr("noise", "cspad-noise.dat");
  
  // initialize arrays
  std::fill_n(&m_count[0], int(MaxQuads), 0UL);
  std::fill_n(&m_segMask[0], int(MaxQuads), 0U);
  std::fill_n(&m_sum[0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, int64_t(0));
  std::fill_n(&m_sum2[0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, int64_t(0));
}

//--------------
// Destructor --
//--------------
CsPadPedestals::~CsPadPedestals ()
{
}

/// Method which is called at the beginning of the run
void 
CsPadPedestals::beginRun(Event& evt, Env& env)
{
  // Find all configuration objects matching the source address
  // provided in configuration. If there is more than one configuration 
  // object is found then complain and stop.
  
  Source src(configStr("source", "DetInfo(:Cspad)"));
  int count = 0;
  
  // need to know segment mask which is availabale in configuration only
  shared_ptr<Psana::CsPad::ConfigV1> config1 = env.configStore().get(src, &m_src);
  if (config1.get()) {
    for (int i = 0; i < MaxQuads; ++i) {
      m_segMask[i] = config1->asicMask()==1 ? 0x3 : 0xff;
    }
    ++ count;
  }
  
  shared_ptr<Psana::CsPad::ConfigV2> config2 = env.configStore().get(src, &m_src);
  if (config2.get()) {
    for (int i = 0; i < MaxQuads; ++i) {
      m_segMask[i] = config2->roiMask(i);
    }
    ++ count;
  }

  shared_ptr<Psana::CsPad::ConfigV3> config3 = env.configStore().get(src, &m_src);
  if (config3.get()) {
    for (int i = 0; i < MaxQuads; ++i) {
      m_segMask[i] = config3->roiMask(i);
    }
    ++ count;
  }

  shared_ptr<Psana::CsPad::ConfigV4> config4 = env.configStore().get(src, &m_src);
  if (config4.get()) {
    for (int i = 0; i < MaxQuads; ++i) {
      m_segMask[i] = config4->roiMask(i);
    }
    ++ count;
  }

  shared_ptr<Psana::CsPad::ConfigV5> config5 = env.configStore().get(src, &m_src);
  if (config5.get()) {
    for (int i = 0; i < MaxQuads; ++i) {
      m_segMask[i] = config5->roiMask(i);
    }
    ++ count;
  }

  if (not count) {
    MsgLog(name(), error, "No CsPad configuration objects found, terminating.");
    terminate();
    return;
  }
  
  if (count > 1) {
    MsgLog(name(), error, "Multiple CsPad configuration objects found, use more specific source address. Terminating.");
    terminate();
    return;
  }

  MsgLog(name(), info, "Found CsPad object with address " << m_src);
  if (m_src.level() != Pds::Level::Source) {
    MsgLog(name(), error, "Found Cspad configuration object with address not at Source level. Terminating.");
    terminate();
    return;
  }

  const Pds::DetInfo& dinfo = static_cast<const Pds::DetInfo&>(m_src);
  // validate that this is indeed cspad, should always be true, but
  // additional protection here should not hurt
  if (dinfo.device() != Pds::DetInfo::Cspad) {
    MsgLog(name(), error, "Found Cspad configuration object with invalid address. Terminating.");
    terminate();
    return;
  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CsPadPedestals::event(Event& evt, Env& env)
{

  // we should get only regular cspad data

  shared_ptr<Psana::CsPad::DataV1> data1 = evt.get(m_src);
  if (data1.get()) {

    int nQuads = data1->quads_shape()[0];
    for (int iq = 0; iq != nQuads; ++ iq) {
      
      // get quad object
      const CsPad::ElementV1& quad = data1->quads(iq);

      // process statistics for this quad
      const ndarray<const int16_t, 3>& data = quad.data();
      collectStat(quad.quad(), data.data());

      ++ m_count[quad.quad()];
    }

  }
  
  shared_ptr<Psana::CsPad::DataV2> data2 = evt.get(m_src);
  if (data2.get()) {

    int nQuads = data2->quads_shape()[0];
    for (int iq = 0; iq != nQuads; ++ iq) {
      
      // get quad object
      const CsPad::ElementV2& quad = data2->quads(iq);

      // process statistics for this quad
      const ndarray<const int16_t, 3>& data = quad.data();
      collectStat(quad.quad(), data.data());

      ++ m_count[quad.quad()];
    }
    
  }

}



/// Method which is called once at the end of the job
void 
CsPadPedestals::endJob(Event& evt, Env& env)
{

  MsgLog(name(), info, "collected total " << m_count[0] << " events");
  
  if (not m_pedFile.empty()) {

    // save pedestals as average
    std::ofstream out(m_pedFile.c_str());
    for (int iq = 0; iq != MaxQuads; ++ iq) {
      for (int is = 0; is != MaxSectors; ++ is) {
        for (int ic = 0; ic != NumColumns; ++ ic) {
          for (int ir = 0; ir != NumRows; ++ ir) {
            double avg = m_count[iq] ? double(m_sum[iq][is][ic][ir]) / m_count[iq] : 0;
            out << avg << ' ';
          }
          out << '\n';
        }
      }
    }

    out.close();

  }

  if (not m_noiseFile.empty()) {
    
    // save pedestals as average
    std::ofstream out(m_noiseFile.c_str());
    for (int iq = 0; iq != MaxQuads; ++ iq) {
      for (int is = 0; is != MaxSectors; ++ is) {
        for (int ic = 0; ic != NumColumns; ++ ic) {
          for (int ir = 0; ir != NumRows; ++ ir) {
            double stdev = 0;
            if (m_count[iq] > 1) {
              stdev = std::sqrt(::stddev2(m_count[iq], m_sum[iq][is][ic][ir], m_sum2[iq][is][ic][ir]));
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


/// collect statistics
void 
CsPadPedestals::collectStat(unsigned qNum, const int16_t* data)
{

  // loop over segments
  int seg = 0;
  for (int is = 0; is < MaxSectors; ++ is) {
    if (m_segMask[qNum] & (1 << is)) {
     
      // beginning of the segment data
      int64_t* sum = &m_sum[qNum][is][0][0];
      int64_t* sum2 = &m_sum2[qNum][is][0][0];

      const int16_t* segData = data + seg*NumColumns*NumRows;

      // sum
      for (int i = 0; i < NumColumns*NumRows ; ++ i) {            
        sum[i] += segData[i];
        sum2[i] += segData[i]*segData[i];
      }          
      
      ++seg;
    }
  }

}

} // namespace cspad_mod
