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
  , m_count(0)
  , m_sum()
  , m_sum2()
{
  m_pedFile = configStr("output", "cspad-pedestals.dat");
  m_noiseFile = configStr("noise", "cspad-noise.dat");
  
  // initialize arrays
  std::fill_n(&m_segMask[0], int(MaxQuads), 0U);
  std::fill_n(&m_sum[0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 0.);
  std::fill_n(&m_sum2[0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 0.);
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
  
  std::string src = configStr("source", "DetInfo()");
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
  if (m_src.level() == Pds::Level::Source) {
    const Pds::DetInfo& dinfo =static_cast<const Pds::DetInfo&>(m_src);
    // see what data we should get
    m_2x2 = dinfo.device() == Pds::DetInfo::Cspad2x2; 
  } else {
    MsgLog(name(), error, "Found object with address not at Source level. Terminating.");
    terminate();
  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CsPadPedestals::event(Event& evt, Env& env)
{

  if (m_2x2) {
    
    // we should expect 2x2 data 
    
    shared_ptr<Psana::CsPad::MiniElementV1> data1 = evt.get(m_src);
    if (data1.get()) {
  
      ++ m_count;
      
      // process statistics for 2x2
      const ndarray<int16_t, 3>& data = data1->data();
      collectStat2x2(data.data());
      
    }
    
  } else {
  
    // we should get only regular cspad data
    
    shared_ptr<Psana::CsPad::DataV1> data1 = evt.get(m_src);
    if (data1.get()) {
  
      ++ m_count;
      
      int nQuads = data1->quads_shape()[0];
      for (int iq = 0; iq != nQuads; ++ iq) {
        
        // get quad object
        const CsPad::ElementV1& quad = data1->quads(iq);
  
        // process statistics for this quad
        const ndarray<int16_t, 3>& data = quad.data();
        collectStat(quad.quad(), data.data());
      }
      
    }
    
    shared_ptr<Psana::CsPad::DataV2> data2 = evt.get(m_src);
    if (data2.get()) {
  
      ++ m_count;
      
      int nQuads = data2->quads_shape()[0];
      for (int iq = 0; iq != nQuads; ++ iq) {
        
        // get quad object
        const CsPad::ElementV2& quad = data2->quads(iq);
  
        // process statistics for this quad
        const ndarray<int16_t, 3>& data = quad.data();
        collectStat(quad.quad(), data.data());
      }
      
    }

  }
}



/// Method which is called once at the end of the job
void 
CsPadPedestals::endJob(Event& evt, Env& env)
{

  MsgLog(name(), info, "collected total " << m_count << " events");
  
  if (m_2x2) {

    const double* sum = &m_sum[0][0][0][0];
    const double* sum2 = &m_sum2[0][0][0][0];
    const int size = NumColumns*NumRows*2;

    if (not m_pedFile.empty()) {
      
      // save pedestals as average
      std::ofstream out(m_pedFile.c_str());
      for (int i = 0; i < size; ++ i) {
        double avg = m_count ? sum[i] / m_count : 0;
        out << avg << '\n';
      }
      
      out.close();
      
    }
  
    if (not m_noiseFile.empty()) {
      
      // save pedestals as average
      std::ofstream out(m_noiseFile.c_str());
      for (int i = 0; i < size; ++ i) {
        double stdev = 0;
        if (m_count > 1) {
          double avg = sum[i] / m_count;
          stdev = std::sqrt(sum2[i] / m_count - avg*avg);              
        }
        out << stdev << '\n';
      }
      
      out.close();
      
    }


  } else {
    
    if (not m_pedFile.empty()) {
      
      // save pedestals as average
      std::ofstream out(m_pedFile.c_str());
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
  
    if (not m_noiseFile.empty()) {
      
      // save pedestals as average
      std::ofstream out(m_noiseFile.c_str());
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
      double* sum = &m_sum[qNum][is][0][0];
      double* sum2 = &m_sum2[qNum][is][0][0];

      const int16_t* segData = data + seg*NumColumns*NumRows;

      // sum
      for (int i = 0; i < NumColumns*NumRows ; ++ i) {            
        sum[i] += double(segData[i]);
        sum2[i] += double(segData[i])*double(segData[i]);
      }          
      
      ++seg;
    }
  }

}

/// collect statistics for 2x2
void 
CsPadPedestals::collectStat2x2(const int16_t* data)
{
  double* sum = &m_sum[0][0][0][0];
  double* sum2 = &m_sum2[0][0][0][0];
  const int size = NumColumns*NumRows*2;
  
  for (int i = 0; i < size; ++ i) {
    double val = data[i];
    sum[i] += val;
    sum2[i] += val*val;
  }          

}

} // namespace cspad_mod
