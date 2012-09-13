//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ProcessAcqiris...
//
// Author List:
//      Ingrid Ofte
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XCorrAnalysis/ProcessAcqiris.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

// Include detector data headers from psddl_psana package:
#include "psddl_psana/acqiris.ddl.h"

#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace XCorrAnalysis;
PSANA_MODULE_FACTORY(ProcessAcqiris)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XCorrAnalysis {

//----------------
// Constructors --
//----------------
ProcessAcqiris::ProcessAcqiris (const std::string& name)
  : Module(name)
  , m_acqSrc()
{
  m_DiodeStart = config("DiodeStart",-1);  // 27400  // Use these to set the integration window for each Diode spectra
  m_DiodeLength = config("DiodeLength",-1);  // 4000
}

//--------------
// Destructor --
//--------------
ProcessAcqiris::~ProcessAcqiris ()
{
}

/// Method which is called once at the beginning of the job
void 
ProcessAcqiris::beginJob(Event& evt, Env& env)
{
  m_count = 0;

}

/// Method which is called at the beginning of the run
void 
ProcessAcqiris::beginRun(Event& evt, Env& env)
{
  Source acqSrc = configStr("acqiris_source","DetInfo(:Acqiris)");
  shared_ptr<Psana::Acqiris::ConfigV1> acqConfig = env.configStore().get(acqSrc, &m_acqSrc);
  MsgLog(name(), info, "Looking for " << acqSrc.src() << " found " << m_acqSrc);
  
  if (acqConfig.get()) {
    m_nch = acqConfig->nbrChannels();

    const Psana::Acqiris::HorizV1& h = acqConfig->horiz();
    if ( m_DiodeStart < 0 ) m_DiodeStart = 0;
    if ( m_DiodeLength < 0 ) m_DiodeLength = h.nbrSamples()*h.nbrSegments();
    
    for (int j(0);j<m_nch; j++){

      for (int i(0);i<MaxSpectra; i++){
	m_AcqDiodeSpect[j][i] = 0.0;
      }
      for (int i(0);i<MaxEvents; i++){
      	m_AcqDiodeArray[j][i] = 0.0;
      }
    }
    
    WithMsgLog(name(), info, str) {
      str << "Acqiris::ConfigV1: nbrBanks=" << acqConfig->nbrBanks()
          << " channelMask=" << acqConfig->channelMask()
          << " nbrChannels=" << acqConfig->nbrChannels()
          << " nbrConvertersPerChannel=" << acqConfig->nbrConvertersPerChannel();
      
      str << "\n  horiz: sampInterval=" << h.sampInterval()
	  << " delayTime=" << h.delayTime()
	  << " nbrSegments=" << h.nbrSegments()
	  << " nbrSamples=" << h.nbrSamples();
      
      const ndarray<Psana::Acqiris::VertV1, 1>& vert = acqConfig->vert();
      for (unsigned ch = 0; ch < vert.shape()[0]; ++ ch) {
        const Psana::Acqiris::VertV1& v = vert[ch];
        str << "\n  vert(" << ch << "):"
            << " fullScale=" << v.fullScale()
            << " slope=" << v.slope()
            << " offset=" << v.offset()
            << " coupling=" << v.coupling()
            << " bandwidth=" << v.bandwidth();
      }
    } 
  }
}

/// Method which is called at the beginning of the calibration cycle
void 
ProcessAcqiris::beginCalibCycle(Event& evt, Env& env)
{
}
  
/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ProcessAcqiris::event(Event& evt, Env& env)
{
  shared_ptr<Psana::Acqiris::DataDescV1> acqData = evt.get(m_acqSrc);
  if (acqData.get()) {
    
    // find matching config object
    shared_ptr<Psana::Acqiris::ConfigV1> acqConfig = env.configStore().get(m_acqSrc);
    
    // loop over channels
    int nchan = acqData->data_shape()[0];
    for (int chan = 0; chan < nchan; ++ chan) {
      
      const Psana::Acqiris::DataDescV1Elem& elem = acqData->data(chan);

      const Psana::Acqiris::VertV1& v = acqConfig->vert()[chan];
      double slope = v.slope();
      double offset = v.offset();

      WithMsgLog(name(), debug, str ) {

        str << "Acqiris::DataDescV1: channel=" << chan
           << "\n  nbrSegments=" << elem.nbrSegments()
           << "\n  nbrSamplesInSeg=" << elem.nbrSamplesInSeg()
           << "\n  indexFirstPoint=" << elem.indexFirstPoint();

        const ndarray<Psana::Acqiris::TimestampV1, 1>& timestamps = elem.timestamp();
        const ndarray<int16_t, 2>& waveforms = elem.waveforms();

        // loop over segments
        for (unsigned seg = 0; seg < elem.nbrSegments(); ++ seg) {

          str << "\n  Segment #" << seg
              << "\n    timestamp=" << timestamps[seg].pos()
              << "\n    data=[";

          unsigned size = std::min(elem.nbrSamplesInSeg(), 32U);
          for (unsigned i = 0; i < size; ++ i) {
            str << (waveforms[seg][i]*slope + offset) << ", ";
          }
          str << "...]";

        }
      }

     // work in progress:
      int numSamplesAcq =  elem.nbrSegments() * elem.nbrSamplesInSeg();
      double* spect0;      // this is the waveform (voltage array)
      double* spect_time0; // this is the timestamp array
      const ndarray<Psana::Acqiris::TimestampV1, 1>& timestamps = elem.timestamp();
      const ndarray<int16_t, 2>& waveforms = elem.waveforms();

      if ((m_DiodeStart+m_DiodeLength) > numSamplesAcq)
        {
          printf( "Diode Spectrum Parameters exceed number of Diode Samples\n");
        }
      else
        {
          int DiodeIndex = 0;
          for (int i = m_DiodeStart; i < (m_DiodeStart+m_DiodeLength); i++)
	    {
	      m_AcqDiodeArray[chan][m_count] += waveforms[0][i];
	      //if(PlotDiodeFlag == 1) {
	      m_AcqDiodeSpect[chan][DiodeIndex] += waveforms[0][i];
	      DiodeIndex++;
	      //}
	    }
	  m_AcqDiodeArray[chan][m_count] /= m_DiodeLength;
        }
      
      
      
      
    }
  }
  
  // increment event counter
  ++ m_count;
}
  
/// Method which is called at the end of the calibration cycle
void 
ProcessAcqiris::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
ProcessAcqiris::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
ProcessAcqiris::endJob(Event& evt, Env& env)
{
}

} // namespace XCorrAnalysis
