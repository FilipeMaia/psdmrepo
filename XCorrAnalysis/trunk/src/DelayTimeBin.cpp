//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DelayTimeBin...
//
// Author List:
//      Ingrid Ofte
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XCorrAnalysis/DelayTimeBin.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <string>
#include <list>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

// Include detector data headers from psddl_psana package:
#include "psddl_psana/encoder.ddl.h"
#include "psddl_psana/bld.ddl.h"

#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace XCorrAnalysis;
PSANA_MODULE_FACTORY(DelayTimeBin)


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XCorrAnalysis {

//----------------
// Constructors --
//----------------
DelayTimeBin::DelayTimeBin (const std::string& name)
  : Module(name)
  , m_encSrc()
  , m_pcSrc()
{
  // get the values from configuration or use defaults
  m_xcorrSrc = configStr("CrossCorrTime","XCorrTimingTool:time");
  m_Delay_a = config("Delay_a", -80.0e-6);  // These are the parameters to convert the encoder position to picoseconds
  m_Delay_b = config("Delay_b", 0.52168);
  m_Delay_c = config("Delay_c",299792458);
  m_Delay_0 = config("Delay_0", 0.0);
  m_NumberBins = config("NumberBins", 300);
  m_StartTime = config("StartTime", 91.0);
  m_EndTime = config("EndTime", 215.0);
  m_DelayTimeFlag = config("DelayTimeFlag", 2); // Set to 0 to not correct timing jitter at all;
  //                                                      1 to correct jitter with phase cavity;
  //                                                      2 to use cross correlator
  m_MaxEvents = config("MaxEvents", 1000000);
}


//--------------
// Destructor --
//--------------
DelayTimeBin::~DelayTimeBin ()
{
}

/// Method which is called once at the beginning of the job
void 
DelayTimeBin::beginJob(Event& evt, Env& env)
{
  m_count=0;
  m_nBadDelayTimes = 0;

  m_DeltaTime = (m_EndTime-m_StartTime)/m_NumberBins;


  // Print run-time parameters
  WithMsgLog(name(), info, str) {
    str << ""
	<< "\n CrossCorrTime = " <<  m_xcorrSrc
	<< "\n Delay_a = "  << m_Delay_a 
	<< "\n Delay_b = "  << m_Delay_b 
	<< "\n Delay_c = "  << m_Delay_c 
	<< "\n Delay_0 = "  << m_Delay_0 ;
  }


}

/// Method which is called at the beginning of the run
void 
DelayTimeBin::beginRun(Event& evt, Env& env)
{

}

/// Method which is called at the beginning of the calibration cycle
void 
DelayTimeBin::beginCalibCycle(Event& evt, Env& env)
{
  Source encSrc = configStr("encSrc","DetInfo(:Encoder)");
  shared_ptr<Psana::Encoder::ConfigV2> config = env.configStore().get(encSrc, &m_encSrc);
  //std::cout << "Looking for " << &encSrc << ", found " << &m_encSrc << std::endl;
  if (config.get()) {
    
    // Detemine from bitmask which channel was in use.
    int bit = 0;    
    while ( (config->chan_mask() & (1 << bit))==0 ) bit++;
    m_encoder_channel = bit;

    WithMsgLog(name(), info, str) {
      str << "Encoder::ConfigV2:";
      str << "\n  chan_mask = " << config->chan_mask();
      str << "\n  count_mode = " << config->count_mode();
      str << "\n  quadrature_mode = " << config->quadrature_mode();
      str << "\n  input_num = " << config->input_num();
      str << "\n  input_rising = " << config->input_rising();
      str << "\n  ticks_per_sec = " << config->ticks_per_sec();
    }
    
  }
  
  Source pcSrc = configStr("pcSrc", "BldInfo(PhaseCavity)");
  shared_ptr<Psana::Bld::BldDataPhaseCavity> cav = env.configStore().get(pcSrc, &m_pcSrc);
  if (cav.get()) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataPhaseCavity:"
          << "\n  fitTime1=" << cav->fitTime1()
          << "\n  fitTime2=" << cav->fitTime2()
          << "\n  charge1=" << cav->charge1()
          << "\n  charge2=" << cav->charge2();
    }
    
  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DelayTimeBin::event(Event& evt, Env& env)
{

  //  Get Encoder to determine DelayTime and BinNumber
  int Encoder_int = 0;

  shared_ptr<Psana::Encoder::DataV2> myEncoder = evt.get(m_encSrc);

  if (myEncoder.get()) {
    Encoder_int = myEncoder->value(m_encoder_channel);
    
    if(Encoder_int > 5e6) Encoder_int -= 2 << 24;
    m_EncoderArray[m_count] = static_cast<double>(Encoder_int);
  }


  //  Get Phase Cavity to determine DelayTime and BinNumber
  shared_ptr<Psana::Bld::BldDataPhaseCavity> cav = evt.get(m_pcSrc);
  double phasecav1 = 0.0; //This is the magic number

  if (cav.get()) {
    phasecav1 = cav->fitTime1();//This is the magic number

    m_PhaseCavityArray[0][m_count] = cav->fitTime1();
    m_PhaseCavityArray[1][m_count] = cav->fitTime2();
    m_PhaseCavityArray[2][m_count] = cav->charge1();
    m_PhaseCavityArray[3][m_count] = cav->charge2();
  }
  
  

  //  Calculate DelayTime and BinNumber
  double DelayTime = 1000000;
  double BinNumber = 0;
  int BinNumber_int = 0;

  if ((myEncoder.get())){

    // Convert Encoder position to delay time in pico seconds:
    double encoderPS =  2.0 * ((m_Delay_a * static_cast<double>(Encoder_int) + m_Delay_b)
			       *1.e-3 / m_Delay_c) / 1.e-12 - m_Delay_0;
    
    // Default delay time (no jitter correction):
    DelayTime = encoderPS;

    // Correct using phase cavity
    if (m_DelayTimeFlag == 1 ){
      DelayTime = encoderPS + phasecav1;
    }

    // Correct using Sanne's Cross-correlator
    else if (m_DelayTimeFlag == 2 ){
      shared_ptr<float> ccCorrTime = evt.get(m_xcorrSrc);
      if ( ccCorrTime.get() ){
	DelayTime = encoderPS + *ccCorrTime;
      }
    }
    
    
    m_DelayTimeArray[m_count] = DelayTime;
    if(DelayTime >= m_StartTime && DelayTime <= m_EndTime)
      {
	BinNumber = ((DelayTime-m_StartTime)/m_DeltaTime);
	BinNumber_int = static_cast<int>(BinNumber);
      }
    else
      {
	m_nBadDelayTimes++;
	//printf("DelayTime out of range. EventIndex = %d\n", m_count);
      }
  }
  else
    {
      printf( "Unable to calculate DelayTime and BinNumber\n");
    }

  // put DelayTime and BinNumber back into the event
  shared_ptr<double> delaytime_ptr(new double(DelayTime));
  shared_ptr<int> binnumber_ptr(new int(BinNumber_int));
  evt.put(delaytime_ptr,"DelayTimeBin:DelayTime");
  evt.put(delaytime_ptr,"DelayTimeBin:BinNumber");
  
  
  // increment event counter
  ++ m_count;
}
  
/// Method which is called at the end of the calibration cycle
void 
DelayTimeBin::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
DelayTimeBin::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
DelayTimeBin::endJob(Event& evt, Env& env)
{
}

} // namespace XCorrAnalysis
