//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: AcqirisArrProducer.cpp 0 2013-10-24 09:00:00Z dubrovin@slac.stanford.edu$
//
// Description:
//	Class AcqirisArrProducer...
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/AcqirisArrProducer.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "pdsdata/xtc/DetInfo.hh"
#include "psddl_psana/acqiris.ddl.h"
#include "ImgAlgos/GlobalMethods.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace std;
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(AcqirisArrProducer)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
AcqirisArrProducer::AcqirisArrProducer (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key_in()
  , m_key_wform()
  , m_key_wtime()
  , m_fname_prefix()
  , m_correct_t()
  , m_print_bits()
  , m_count_event(0)
  , m_count_calib(0)
{
  m_str_src           = configSrc("source",  "DetInfo(:Acqiris)");
  m_key_in            = configStr("key_in",                   "");
  m_key_wform         = configStr("key_wform",       "acq_wform");
  m_key_wtime         = configStr("key_wtime",       "acq_wtime");
  m_fname_prefix      = configStr("fname_prefix",             "");
  m_correct_t         = config   ("correct_t",             true );
  m_print_bits        = config   ("print_bits",               0 );

  m_do_save_config = (m_fname_prefix.empty()) ? false : true;

  if( m_print_bits & 1 ) printInputParameters();
}

//--------------------

void 
AcqirisArrProducer::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters :"
        << "\n source           : " << m_str_src
        << "\n key_in           : " << m_key_in      
        << "\n key_wform        : " << m_key_wform
        << "\n key_wtime        : " << m_key_wtime
        << "\n fname_prefix     : " << m_fname_prefix
        << "\n correct_t        : " << m_correct_t
        << "\n print_bits       : " << m_print_bits
        << "\n do_save_config   : " << m_do_save_config
        << "\n";     
       }
}


//--------------
// Destructor --
//--------------
AcqirisArrProducer::~AcqirisArrProducer ()
{
}

void 
AcqirisArrProducer::beginJob(Event& evt, Env& env)
{
}

void 
AcqirisArrProducer::beginRun(Event& evt, Env& env)
{
  m_str_runnum     = stringRunNumber(evt);
  m_str_experiment = stringExperiment(env);
  m_fname_common   = m_fname_prefix + "-" + m_str_experiment + "-r" + m_str_runnum;
}

void 
AcqirisArrProducer::beginCalibCycle(Event& evt, Env& env)
{
  m_count_calib ++;

}

//-------------------------------
// Method which is called with event data
void 
AcqirisArrProducer::event(Event& evt, Env& env)
{
  m_count_event ++;
  if (m_count_event == 1) {
    std::string txt_config = "\n" + getAcqirisConfig(evt, env);
    if( m_print_bits & 2 ) MsgLog(name(), info, txt_config);
    if( m_do_save_config ) saveTextInFile(m_fname_common + "-config.txt", txt_config, m_print_bits & 4);
  }

  if( m_print_bits & 8 ) print_wf_in_event(evt, env);
  proc_and_put_wf_in_event(evt, env);
}

//-------------------------------

std::string
AcqirisArrProducer::getAcqirisConfig(Event& evt, Env& env)
{
  shared_ptr<Psana::Acqiris::ConfigV1> acqConfig = env.configStore().get(m_str_src, &m_src);
  if (acqConfig) {
      stringstream ss; 
      ss  << "Acqiris::ConfigV1:\n"
          << "  nbrBanks="    << acqConfig->nbrBanks()
          << " channelMask="  << acqConfig->channelMask()
          << " nbrChannels="  << acqConfig->nbrChannels()
          << " nbrConvertersPerChannel=" << acqConfig->nbrConvertersPerChannel();
     
      const Psana::Acqiris::HorizV1& h = acqConfig->horiz();
      ss   << "\n  horiz: sampInterval=" << h.sampInterval()
           << " delayTime="              << h.delayTime()
           << " nbrSegments="            << h.nbrSegments()
           << " nbrSamples="             << h.nbrSamples();
      
      const ndarray<const Psana::Acqiris::VertV1, 1>& vert = acqConfig->vert();
      for (unsigned ch = 0; ch < acqConfig->nbrChannels(); ++ ch) {
        const Psana::Acqiris::VertV1& v = vert[ch];
        ss  << "\n  vert(" << ch << "):"
            << " fullScale="  << v.fullScale()
            << " slope="      << v.slope()
            << " offset="     << v.offset()
            << " coupling="   << v.coupling()
            << " bandwidth="  << v.bandwidth();
      }

      return ss.str();
  }
  
  return std::string("WARNING! Acqiris::ConfigV1 is not forund...");
}

//-------------------------------

void 
AcqirisArrProducer::print_wf_in_event(Event& evt, Env& env)
{

  //Pds::Src src;
  shared_ptr<Psana::Acqiris::DataDescV1> acqData = evt.get(m_src);
  if (acqData) {
    
    // find matching config object
    shared_ptr<Psana::Acqiris::ConfigV1> acqConfig = env.configStore().get(m_src);
    
    // loop over channels
    int nchan = acqData->data_shape()[0];
    for (int chan = 0; chan < nchan; ++ chan) {
      
      const Psana::Acqiris::DataDescV1Elem& elem = acqData->data(chan);

      const Psana::Acqiris::VertV1& v = acqConfig->vert()[chan];
      double slope = v.slope();
      double offset = v.offset();

      WithMsgLog(name(), info, str ) {

        str << "Acqiris::DataDescV1: channel=" << chan
           << "\n  nbrSegments="     << elem.nbrSegments()
           << "\n  nbrSamplesInSeg=" << elem.nbrSamplesInSeg()
           << "\n  indexFirstPoint=" << elem.indexFirstPoint();
        
        const ndarray<const Psana::Acqiris::TimestampV1, 1>& timestamps = elem.timestamp();
        const ndarray<const int16_t, 2>& waveforms = elem.waveforms();

        // loop over segments
        for (unsigned seg = 0; seg < elem.nbrSegments(); ++ seg) {

          unsigned size = elem.nbrSamplesInSeg();
          ndarray<const int16_t, 1> raw(waveforms[seg]);
          ndarray<float, 1> wf = make_ndarray<float>(size);
          for (unsigned i = 0; i < size; ++ i) {
            wf[i] = raw[i]*slope - offset;
          }
          
          str << "\n  Segment #" << seg
              << "\n    timestamp value = " << timestamps[seg].value() << " pos = " << timestamps[seg].pos()
              << "\n    raw = " << raw
              << "\n    data = " << wf;
        
        }
      }
    }
  }
}
//-------------------------------

void 
AcqirisArrProducer::print_wf_index_info(uint32_t indexFirstPoint, int32_t i0_seg, int32_t size)
{
  MsgLog(name(), info, "event = " << m_count_event
	               << "   indexFirstPoint = " << indexFirstPoint   // 0,1,2,3
	               << "   i0_seg = " << i0_seg	                 // always 0 in my test
		       << "   iterator size = " << size);  
}

//-------------------------------

void 
AcqirisArrProducer::proc_and_put_wf_in_event(Event& evt, Env& env)
{
  shared_ptr<Psana::Acqiris::DataDescV1> acqData = evt.get(m_src);
  if (acqData) {
    // find matching config object
    shared_ptr<Psana::Acqiris::ConfigV1> acqConfig = env.configStore().get(m_src);
    // int nbrChannels = acqData->data_shape()[0];
    //uint32_t channelMask = acqConfig->channelMask();
    uint32_t nbrChannels = acqConfig->nbrChannels();
    
    const Psana::Acqiris::HorizV1& h = acqConfig->horiz();
    double sampInterval = h.sampInterval();
    uint32_t nbrSamples = h.nbrSamples();


    //const unsigned shape[] = {nbrChannels, nbrSamples};
    ndarray<wform_t, 2> wf = make_ndarray<wform_t>(nbrChannels, nbrSamples);
    ndarray<wtime_t, 2> wt = make_ndarray<wtime_t>(nbrChannels, nbrSamples);

    std::fill_n(wf.data(), int(nbrChannels*nbrSamples), double(0.));
    std::fill_n(wt.data(), int(nbrChannels*nbrSamples), double(0.));

    // loop over channels
    for (uint32_t chan = 0; chan < nbrChannels; ++ chan) {
      
        const Psana::Acqiris::DataDescV1Elem& elem = acqData->data(chan);

        const Psana::Acqiris::VertV1& v = acqConfig->vert()[chan];
        double slope  = v.slope();
        double offset = v.offset();

        uint32_t nbrSegments     = elem.nbrSegments();
        uint32_t indexFirstPoint = elem.indexFirstPoint();
        uint32_t nbrSamplesInSeg = elem.nbrSamplesInSeg();
        
        const ndarray<const Psana::Acqiris::TimestampV1, 1>& timestamps = elem.timestamp();
        const ndarray<const int16_t, 2>& waveforms = elem.waveforms();

        // loop over segments WITH time correction
        for (unsigned seg = 0; seg < nbrSegments; ++ seg) {

            ndarray<const int16_t, 1> raw(waveforms[seg]);
	  //double timestamp = timestamps[seg].value();
            double pos       = timestamps[seg].pos();        

       	    int32_t i0_seg = (m_correct_t) ? seg * nbrSamplesInSeg + int32_t(pos/sampInterval) : seg * nbrSamplesInSeg;

	    // Protection against aray index overflow
	    int32_t size = (i0_seg + nbrSamplesInSeg <= nbrSamples) ? nbrSamplesInSeg : nbrSamples - i0_seg;

       	    if (m_correct_t) {
	      // Protection against aray index overflow
	      if( indexFirstPoint + size > nbrSamplesInSeg ) size = nbrSamplesInSeg - indexFirstPoint;	    
              if( m_print_bits & 16 ) print_wf_index_info(indexFirstPoint, i0_seg, size);
	    
              for (int32_t i = 0; i < size; ++ i) {
	        wf[chan][i0_seg + i] = raw[indexFirstPoint + i]*slope - offset;
                wt[chan][i0_seg + i] = i*sampInterval + pos;		
              }          	    
	    
	    } else {	    

              if( m_print_bits & 4 ) print_wf_index_info(indexFirstPoint, i0_seg, size);
	    
              for (int32_t i = 0; i < size; ++ i) {
	        wf[chan][i0_seg + i] = raw[i]*slope - offset;
                wt[chan][i0_seg + i] = i*sampInterval + pos;		
              }          

            } // if (m_correct_t) 

  	} // for (unsigned seg 

    } // loop over channels

    saveNonConst2DArrayInEvent<wform_t> (evt, m_src, m_key_wform, wf);
    saveNonConst2DArrayInEvent<wtime_t> (evt, m_src, m_key_wtime, wt);
  } // if (acqData)
}

} // namespace ImgAlgos
//-------------------------------
//-------------------------------
//-------------------------------
