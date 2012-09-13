//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XCorrMakeBkg...
//
// Author List:
//      Sanne de Jong  (Original)
//      Ingrid Ofte    (Adaption to the psana framework)
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XCorrAnalysis/XCorrMakeBkg.h"

//-----------------
// C/C++ Headers --
//-----------------
//#include <sys/dir.h>
//#include <sys/types.h>
#include <sys/stat.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

// Include detector data headers from psddl_psana package:
#include "psddl_psana/opal1k.ddl.h"
#include "psddl_psana/camera.ddl.h"

#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace XCorrAnalysis;
PSANA_MODULE_FACTORY(XCorrMakeBkg)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XCorrAnalysis {

//----------------
// Constructors --
//----------------
XCorrMakeBkg::XCorrMakeBkg (const std::string& name)
  : Module(name)
  , m_opalSrc()
  , m_darkFileOut()
{
  // get the values from configuration or use defaults
  m_opalSrc = configStr("opalSrc", "DetInfo(:Opal1000)");  
  m_darkFileOut = configStr("darkFileOut","darkout.txt");
}

//--------------
// Destructor --
//--------------
XCorrMakeBkg::~XCorrMakeBkg ()
{
}

/// Method which is called once at the beginning of the job
void 
XCorrMakeBkg::beginJob(Event& evt, Env& env)
{
  m_count = 0;
}

/// Method which is called at the beginning of the run
void 
XCorrMakeBkg::beginRun(Event& evt, Env& env)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    m_runNr = eventId->run();
  }

}

/// Method which is called at the beginning of the calibration cycle
void 
XCorrMakeBkg::beginCalibCycle(Event& evt, Env& env)
{
  shared_ptr<Psana::Opal1k::ConfigV1> config = env.configStore().get(m_opalSrc);
  if (config.get()) {
    
    WithMsgLog(name(), info, str) {
      str << "Psana::Opal1k::ConfigV1:";
      str << "\n  black_level = " << config->black_level();
      str << "\n  gain_percent = " << config->gain_percent();
      str << "\n  output_resolution = " << config->output_resolution();
      str << "\n  vertical_binning = " << config->vertical_binning();
      str << "\n  output_mirroring = " << config->output_mirroring();
      str << "\n  vertical_remapping = " << int(config->vertical_remapping());
      str << "\n  output_offset = " << config->output_offset();
      str << "\n  output_resolution_bits = " << config->output_resolution_bits();
      str << "\n  defect_pixel_correction_enabled = " << int(config->defect_pixel_correction_enabled());
      str << "\n  output_lookup_table_enabled = " << int(config->output_lookup_table_enabled());

      if (config->output_lookup_table_enabled()) {
        const ndarray<uint16_t, 1>& output_lookup_table = config->output_lookup_table();
        str << "\n  output_lookup_table =";
        for (unsigned i = 0; i < output_lookup_table.size(); ++ i) {
          str << ' ' << output_lookup_table[i];
        }

      }


      if (config->number_of_defect_pixels()) {
        str << "\n  defect_pixel_coordinates =";
        const ndarray<Psana::Camera::FrameCoord, 1>& coord = config->defect_pixel_coordinates();
        for (unsigned i = 0; i < coord.size(); ++ i) {
	  str << "(" << coord[i].column() << ", " << coord[i].row() << ")";
        }
      }
    }
  }

  
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
XCorrMakeBkg::event(Event& evt, Env& env)
{
//  Process Opal Frame
  shared_ptr<Psana::Camera::FrameV1> opal = evt.get(m_opalSrc);
  if (opal.get()) {

    const ndarray<uint16_t, 2>& data = opal->data16();
    long ccsize = data.size();

    if (0){
      WithMsgLog(name(), info, str) {
	str << "\n  data =";
	for (int i = 0; i < 10; ++ i) {
	  str << " " << data[0][i];
	}
	str << " ...";
      }
    }

    if (!m_ccaverage.get() ){
      MsgLog(name(),info,"First event, data shape " << data.shape()[0] << " " << data.shape()[1]);

      //*m_ccaverage = new ndarray<uint16_t,2>(data);
      
      // finally: this works, but it's not a shared_ptr
      //ndarray<uint16_t,2> *ccaverage = new ndarray<uint16_t,2>(data);
      // and this doesn't work .. why? 
      shared_ptr< ndarray<uint16_t,2> > m_ccaverage( new ndarray<uint16_t,2>(data) );
    } else {

      MsgLog(name(),info,"Another event, data shape " << data.shape()[0] << " " << data.shape()[1]);

      for(int i =0 ; i < ccsize ; i++){
	//m_ccaverage[i] += data[i];
      }
    }
    
  } else {
    std::cout << "No opal " << std::endl;
  }
  
  

  // increment event counter
  ++ m_count;
}
  
/// Method which is called at the end of the calibration cycle
void 
XCorrMakeBkg::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
XCorrMakeBkg::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
XCorrMakeBkg::endJob(Event& evt, Env& env)
{
}

} // namespace XCorrAnalysis


