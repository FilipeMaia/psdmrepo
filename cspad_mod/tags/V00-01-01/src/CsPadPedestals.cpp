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
  std::string src = configStr("source", "DetInfo(:Cspad)");

  // need to know segment mask which is availabale in configuration only
  shared_ptr<Psana::CsPad::ConfigV1> config1 = env.configStore().get(src, &m_src);
  if (config1.get()) {
    for (int i = 0; i < MaxQuads; ++i) {
      m_segMask[i] = config1->asicMask()==1 ? 0x3 : 0xff;
    }
  }
  shared_ptr<Psana::CsPad::ConfigV2> config2 = env.configStore().get(src, &m_src);
  if (config2.get()) {
    for (int i = 0; i < MaxQuads; ++i) {
      m_segMask[i] = config2->roiMask(i);
    }
  }

  shared_ptr<Psana::CsPad::ConfigV3> config3 = env.configStore().get(src, &m_src);
  if (config3.get()) {
    for (int i = 0; i < MaxQuads; ++i) {
      m_segMask[i] = config3->roiMask(i);
    }
  }
  
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CsPadPedestals::event(Event& evt, Env& env)
{
  
  shared_ptr<Psana::CsPad::DataV1> data1 = evt.get(m_src);
  if (data1.get()) {

    ++ m_count;
    
    int nQuads = data1->quads_shape()[0];
    for (int iq = 0; iq != nQuads; ++ iq) {
      
      // get quad object
      const CsPad::ElementV1& quad = data1->quads(iq);

      // process statistics for this quad
      collectStat(quad.quad(), quad.data());
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
      collectStat(quad.quad(), quad.data());
    }
    
  }
  
}



/// Method which is called once at the end of the job
void 
CsPadPedestals::endJob(Event& evt, Env& env)
{

  MsgLog(name(), info, "collected total " << m_count << " events");
  
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

} // namespace cspad_mod
