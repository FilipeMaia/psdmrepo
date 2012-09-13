//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ProcessFccd...
//
// Author List:
//      Ingrid Ofte
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XCorrAnalysis/ProcessFccd.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"

// Include detector data headers from psddl_psana package:
#include "psddl_psana/evr.ddl.h"
#include "psddl_psana/fccd.ddl.h"
#include "psddl_psana/camera.ddl.h"

#include <list>

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace XCorrAnalysis; // should be the name of the package (?)
PSANA_MODULE_FACTORY(ProcessFccd)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XCorrAnalysis {

//----------------
// Constructors --
//----------------
ProcessFccd::ProcessFccd (const std::string& name)
  : Module(name)
  , m_src()
{
  // get the values from configuration or use defaults
  m_src = configStr("detector name", "DetInfo(:Fccd)");
  m_img_out = configStr("image_out","MyFCCD");
  m_BackgroundParam = config("BackgroundParameter",3);
  // Set to 1 to use the DarkCounts file or 
  // 2 to use integrated region for each image or 
  // 3 to use both
  m_pkiRegionStart = config("pkiRegionStart", 50);// PeakIntensity
  m_pkiRegionEnd = config("pkiRegionEnd",225);
  m_bgiRegionStart = config("bgiRegionStart", 150); // BackgroundIntensity
  m_bgiRegionEnd = config("bgiRegionEnd",320);
}
  
//--------------
// Destructor --
//--------------
ProcessFccd::~ProcessFccd ()
{
}

/// Method which is called once at the beginning of the job
void 
ProcessFccd::beginJob(Event& evt, Env& env)
{
  m_count = 0;
}

/// Method which is called at the beginning of the run
void 
ProcessFccd::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
ProcessFccd::beginCalibCycle(Event& evt, Env& env)
{
  shared_ptr<Psana::FCCD::FccdConfigV2> config2 = env.configStore().get(m_src);
  if (config2.get()) {
    
    WithMsgLog(name(), info, str) {
      str << "FCCD::FccdConfigV2:";
      str << "\n  outputMode = " << config2->outputMode();
      str << "\n  ccdEnable = " << int(config2->ccdEnable());
      str << "\n  focusMode = " << int(config2->focusMode());
      str << "\n  exposureTime = " << config2->exposureTime();
      str << "\n  dacVoltages = [" << config2->dacVoltages()[0]
          << " " << config2->dacVoltages()[1] << " ...]";
      str << "\n  waveforms = [" << config2->waveforms()[0]
          << " " << config2->waveforms()[1] << " ...]";
      str << "\n  width = " << config2->width();
      str << "\n  height = " << config2->height();
      str << "\n  trimmedWidth = " << config2->trimmedWidth();
      str << "\n  trimmedHeight = " << config2->trimmedHeight();
    }

  }
  
  
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ProcessFccd::event(Event& evt, Env& env)
{
  // example of getting non-detector data from event
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    // example of producing messages using MgsLog facility
    MsgLog(name(), info, "event ID: " << *eventId);
  }
  
  //  Process FCCD Frame
  shared_ptr<Psana::Camera::FrameV1> fccd = evt.get(m_src);
  if (fccd.get()) {
    const ndarray<uint8_t, 2>& img8 = fccd->data8();

    // make ndarray image
    m_raw_image = make_ndarray((uint16_t*)img8.data(), img8.shape()[0], img8.shape()[1]/2 );
    

    // Remove Overscanlines from Raw Image
    for(int n=1;n<=48;n++)
      {
	for(int i=0;i<=239;i++)
	  {
	    {
	      int Index = 0;
	      for(int j=(n-1)*10;j<=9+(n-1)*10;j++)
		{
		  whole_image[i][j] = m_raw_image[i+6][Index+(n-1)*12];
		  whole_image[i+240][j] = m_raw_image[i+253][Index+2+((n-1)*12)];
		  half_image[i][j] = m_raw_image[i+253][Index+2+((n-1)*12)];
		  Index++;
		}
	    }
	  }
      }
    
    
    
    ////////////////////////////////////////////////////////////////
    if(m_BackgroundParam == 3)
      {
	float BackgroundLevel = 0;
	for (int i=m_pkiRegionStart;i<=m_pkiRegionEnd;i++)
	  {
	    for (int j=m_bgiRegionStart;j<=m_bgiRegionEnd;j++)
	      {
		BackgroundLevel += whole_image[i][j];
		//    n++;
	      }
	  }
	//BackgroundLevel /= n;
	//BackgroundArray[1][m_count] = BackgroundLevel;
      }
    ////////////////////////////////

    //image = make_ndarray((uint16_t*)img8.data(), img8.shape()[0], img8.shape()[1]/2 );
    
    ndarray<uint16_t, 2> image = make_ndarray((uint16_t*)0, 480,480);
    std::cout << "image shape " << image.shape()[0] << " " << image.shape()[1] << std::endl;


    shared_ptr< ndarray<uint16_t,2> > image_ptr = boost::make_shared< ndarray<uint16_t,2> >(image);
    evt.put(image_ptr, m_img_out);
    
    
  } else {
    std::cout << "No fccd " << std::endl;
  }

  
  

  //ProcessFccd::QuickTest(evt,env);

  
  // increment event counter
  ++ m_count;

  // ---------------------------------------------
  // give (corrected) image to the event
  
}
  
/// Method which is called at the end of the calibration cycle
void 
ProcessFccd::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
ProcessFccd::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
ProcessFccd::endJob(Event& evt, Env& env)
{
}

void ProcessFccd::QuickTest(Event& evt,Env& env)
{
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << std::endl;
  const EpicsStore& estore = env.epicsStore();
  std::vector<std::string> pvNames = estore.pvNames();
  size_t size = pvNames.size();
  shared_ptr<Psana::Epics::EpicsPvHeader> pv = estore.getPV(pvNames[100]);
  std::cout << pvNames[100] << std::endl;
  std::cout << pv->numElements() << std::endl;
  for (int e = 0; e < pv->numElements(); ++ e) {
    const double& value = estore.value(pvNames[100], e);
    std::cout <<  value << std::endl;
  }
  //std::cout << estore.value(pvNames[100],0) << std::endl;
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << std::endl;
}


} // namespace XCorrAnalysis
