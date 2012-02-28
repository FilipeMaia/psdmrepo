//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2Pedestals...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "cspad_mod/CsPad2x2Pedestals.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <fstream>
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/cspad.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace Psana;

// This declares this class as psana module
using namespace cspad_mod;
PSANA_MODULE_FACTORY(CsPad2x2Pedestals)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace cspad_mod {

//----------------
// Constructors --
//----------------
CsPad2x2Pedestals::CsPad2x2Pedestals (const std::string& name)
  : Module(name)
  , m_pedFile()
  , m_noiseFile()
  , m_src()
  , m_count(0)
  , m_sum()
  , m_sum2()
{
  m_pedFile = configStr("output", "cspad2x2-pedestals.dat");
  m_noiseFile = configStr("noise", "cspad2x2-noise.dat");
  
  // initialize arrays
  std::fill_n(&m_sum[0][0][0], MaxSectors*NumColumns*NumRows, 0.);
  std::fill_n(&m_sum2[0][0][0], MaxSectors*NumColumns*NumRows, 0.);
}

//--------------
// Destructor --
//--------------
CsPad2x2Pedestals::~CsPad2x2Pedestals ()
{
}

/// Method which is called at the beginning of the run
void 
CsPad2x2Pedestals::beginRun(Event& evt, Env& env)
{
  // Find all configuration objects matching the source address
  // provided in configuration. If there is more than one configuration 
  // object is found then complain and stop.
  
  std::string src = configStr("source", "DetInfo(:Cspad2x2)");
  int count = 0;
  
  // cspad2x2 data could come with either CsPad2x2::ConfigV1 or CsPad::ConfigV3
  // configuration (latter happened for brief period)
  shared_ptr<Psana::CsPad2x2::ConfigV1> config1 = env.configStore().get(src, &m_src);
  if (config1.get()) {
    ++ count;
  }
  
  shared_ptr<Psana::CsPad::ConfigV3> config3 = env.configStore().get(src, &m_src);
  if (config3.get()) {
    ++ count;
  }

  if (not count) {
    MsgLog(name(), error, "No CsPad2x2 configuration objects found, terminating.");
    terminate();
    return;
  }
  
  if (count > 1) {
    MsgLog(name(), error, "Multiple CsPad2x2 configuration objects found, use more specific source address. Terminating.");
    terminate();
    return;
  }

  MsgLog(name(), info, "Found CsPad2x2 object with address " << m_src);
  if (m_src.level() != Pds::Level::Source) {
    MsgLog(name(), error, "Found Cspad2x2 configuration object with address not at Source level. Terminating.");
    terminate();
    return;
  }

  const Pds::DetInfo& dinfo = static_cast<const Pds::DetInfo&>(m_src);
  // validate that this is indeed cspad2x2, should always be true, but
  // additional protection here should not hurt
  if (dinfo.device() != Pds::DetInfo::Cspad2x2) {
    MsgLog(name(), error, "Found Cspad2x2 configuration object with invalid address. Terminating.");
    terminate();
    return;
  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CsPad2x2Pedestals::event(Event& evt, Env& env)
{

  // we should expect 2x2 data
  shared_ptr<Psana::CsPad2x2::ElementV1> data1 = evt.get(m_src);
  if (data1.get()) {

    ++ m_count;
    
    // process statistics for 2x2
    const ndarray<int16_t, 3>& data = data1->data();
    collectStat(data.data());
    
  }
    
}



/// Method which is called once at the end of the job
void 
CsPad2x2Pedestals::endJob(Event& evt, Env& env)
{

  MsgLog(name(), info, "collected total " << m_count << " events");
  
  const double* sum = &m_sum[0][0][0];
  const double* sum2 = &m_sum2[0][0][0];
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

}


/// collect statistics for 2x2
void 
CsPad2x2Pedestals::collectStat(const int16_t* data)
{
  double* sum = &m_sum[0][0][0];
  double* sum2 = &m_sum2[0][0][0];
  const int size = NumColumns*NumRows*2;
  
  for (int i = 0; i < size; ++ i) {
    double val = data[i];
    sum[i] += val;
    sum2[i] += val*val;
  }          

}

} // namespace cspad_mod
