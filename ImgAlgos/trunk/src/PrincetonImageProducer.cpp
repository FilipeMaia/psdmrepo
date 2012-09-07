//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonImageProducer...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/PrincetonImageProducer.h"

//-----------------
// C/C++ Headers --
//-----------------
//#include <iomanip> // for setw, setfill
//#include <sstream> // for streamstring
//#include <iostream>// for setf

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/princeton.ddl.h"
#include "ImgAlgos/GlobalMethods.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(PrincetonImageProducer)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
PrincetonImageProducer::PrincetonImageProducer (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key_in()
  , m_key_out() 
//, m_subtract_offset()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_str_src           = configStr("source", "DetInfo(:Princeton)");
  m_key_in            = configStr("key_in",                 "");
  m_key_out           = configStr("key_out",           "image");
//m_subtract_offset   = config   ("subtract_offset",      true);
  m_print_bits        = config   ("print_bits",             0 );
}

//--------------------

void 
PrincetonImageProducer::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters :"
        << "\n source           : " << m_str_src
        << "\n key_in           : " << m_key_in      
        << "\n key_out          : " << m_key_out
      //<< "\n subtract_offset  : " << m_subtract_offset     
        << "\n print_bits       : " << m_print_bits
        << "\n";     
  }
}

//--------------------

//--------------
// Destructor --
//--------------
PrincetonImageProducer::~PrincetonImageProducer ()
{
}

/// Method which is called once at the beginning of the job
void 
PrincetonImageProducer::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
}

/// Method which is called at the beginning of the run
void 
PrincetonImageProducer::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
PrincetonImageProducer::beginCalibCycle(Event& evt, Env& env)
{
  if( m_print_bits & 16 ) {

    MsgLog(name(), info, "in beginCalibCycle()");
    
    shared_ptr<Psana::Princeton::ConfigV1> config1 = env.configStore().get(m_src);
    if (config1.get()) {    
      WithMsgLog(name(), info, str) {
        str << "Princeton::ConfigV1:";
        str << "\n  width = " << config1->width();
        str << "\n  height = " << config1->height();
        str << "\n  orgX = " << config1->orgX();
        str << "\n  orgY = " << config1->orgY();
        str << "\n  binX = " << config1->binX();
        str << "\n  binY = " << config1->binY();
        str << "\n  exposureTime = " << config1->exposureTime();
        str << "\n  coolingTemp = " << config1->coolingTemp();
        str << "\n  readoutSpeedIndex = " << config1->readoutSpeedIndex();
        str << "\n  readoutEventCode = " << config1->readoutEventCode();
        str << "\n  delayMode = " << config1->delayMode();
        str << "\n  frameSize = " << config1->frameSize();
        str << "\n  numPixels = " << config1->numPixels();
      }
    }
    
    shared_ptr<Psana::Princeton::ConfigV2> config2 = env.configStore().get(m_src);
    if (config2.get()) {
      WithMsgLog(name(), info, str) {
        str << "Princeton::ConfigV2:";
        str << "\n  width = " << config2->width();
        str << "\n  height = " << config2->height();
        str << "\n  orgX = " << config2->orgX();
        str << "\n  orgY = " << config2->orgY();
        str << "\n  binX = " << config2->binX();
        str << "\n  binY = " << config2->binY();
        str << "\n  exposureTime = " << config2->exposureTime();
        str << "\n  coolingTemp = " << config2->coolingTemp();
        str << "\n  gainIndex = " << config2->gainIndex();
        str << "\n  readoutSpeedIndex = " << config2->readoutSpeedIndex();
        str << "\n  readoutEventCode = " << config2->readoutEventCode();
        str << "\n  delayMode = " << config2->delayMode();
        str << "\n  frameSize = " << config2->frameSize();
        str << "\n  numPixels = " << config2->numPixels();
      }    
    }
    
    shared_ptr<Psana::Princeton::ConfigV3> config3 = env.configStore().get(m_src);
    if (config3.get()) {    
      WithMsgLog(name(), info, str) {
        str << "Princeton::ConfigV2:";
        str << "\n  width = " << config3->width();
        str << "\n  height = " << config3->height();
        str << "\n  orgX = " << config3->orgX();
        str << "\n  orgY = " << config3->orgY();
        str << "\n  binX = " << config3->binX();
        str << "\n  binY = " << config3->binY();
        str << "\n  exposureTime = " << config3->exposureTime();
        str << "\n  coolingTemp = " << config3->coolingTemp();
        str << "\n  gainIndex = " << config3->gainIndex();
        str << "\n  readoutSpeedIndex = " << config3->readoutSpeedIndex();
        str << "\n  exposureEventCode = " << config3->exposureEventCode();
        str << "\n  numDelayShots = " << config3->numDelayShots();
        str << "\n  frameSize = " << config3->frameSize();
        str << "\n  numPixels = " << config3->numPixels();
      } 
    }
  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
PrincetonImageProducer::event(Event& evt, Env& env)
{
  ++ m_count;
  if( m_print_bits & 2 ) printEventRecord(evt);
  procEvent(evt,env);
}
  
/// Method which is called at the end of the calibration cycle
void 
PrincetonImageProducer::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
PrincetonImageProducer::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
PrincetonImageProducer::endJob(Event& evt, Env& env)
{
  if( m_print_bits & 4 ) printSummary(evt);
}

//--------------------
//--------------------

void 
PrincetonImageProducer::procEvent(Event& evt, Env& env)
{
  shared_ptr<Psana::Princeton::FrameV1> frame = evt.get(m_str_src, m_key_in, &m_src);
  if (frame.get()) {

      const ndarray<uint16_t, 2>& data = frame->data();

      /*
      // copy data with type changing 
      if(m_count == 1) 
        m_data = new double [data.size()];
      unsigned ind = 0;
      ndarray<uint16_t, 2>::const_iterator cit;
      for(cit=data.begin(); cit!=data.end(); cit++) { m_data[ind++] = double(*cit); }
      save2DArrayInEvent<double>   (evt, m_src, m_key_out, m_data, data.shape());
      */
 
      save2DArrayInEvent<uint16_t> (evt, m_src, m_key_out, data.data(), data.shape());

      if( m_print_bits & 8 ) {
        std::cout << "  data =";
        for (int i = 0; i < 10; ++ i) {
          std::cout << " " << data[0][i];
        }
        std::cout << " ...\n";
      }
  }
  else
  {
    const std::string msg = "Princeton::FrameV1 object is not available in the event(...) for source:" 
                          + m_str_src + " key:" + m_key_in;
    MsgLog(name(), info, msg);       
  }
}

//--------------------

void 
PrincetonImageProducer::printEventRecord(Event& evt, std::string comment)
{
  MsgLog( name(), info,  "Run="    << stringRunNumber(evt) 
                     << " Evt="    << stringFromUint(m_count) 
                     << " Time="   << stringTimeStamp(evt) 
	             << comment.c_str() 
  );
}

//--------------------

void 
PrincetonImageProducer::printSummary(Event& evt, std::string comment)
{
  MsgLog( name(), info, "Run=" << stringRunNumber(evt) 
	                << " Number of processed events=" << stringFromUint(m_count)
                        << comment.c_str()
  );
}

//--------------------
} // namespace ImgAlgos
