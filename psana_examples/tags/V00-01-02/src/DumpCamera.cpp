//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpCamera...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpCamera.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/camera.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpCamera)

namespace {
  
  void
  printFrameCoord(std::ostream& str, const Psana::Camera::FrameCoord& coord) 
  {
    str << "(" << coord.column() << ", " << coord.row() << ")";
  }
  
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpCamera::DumpCamera (const std::string& name)
  : Module(name)
{
  m_src = configStr("source", "DetInfo(:Opal1000)");
}

//--------------
// Destructor --
//--------------
DumpCamera::~DumpCamera ()
{
}

// Method which is called at the beginning of the calibration cycle
void 
DumpCamera::beginCalibCycle(Event& evt, Env& env)
{
  MsgLog(name(), trace, ": in beginCalibCycle()");

  shared_ptr<Psana::Camera::FrameFexConfigV1> frmConfig = env.configStore().get(m_src);
  if (not frmConfig.get()) {
    MsgLog(name(), info, "Camera::FrameFexConfigV1 not found");    
  } else {
    
    WithMsgLog(name(), info, str) {
      str << "Camera::FrameFexConfigV1:";
      str << "\n  forwarding = " << frmConfig->forwarding();
      str << "\n  forward_prescale = " << frmConfig->forward_prescale();
      str << "\n  processing = " << frmConfig->processing();
      str << "\n  roiBegin = ";
      ::printFrameCoord(str, frmConfig->roiBegin());
      str << "\n  roiEnd = ";
      ::printFrameCoord(str, frmConfig->roiEnd());
      str << "\n  threshold = " << frmConfig->threshold();
      str << "\n  number_of_masked_pixels = " << frmConfig->number_of_masked_pixels();
      for (unsigned i = 0; i < frmConfig->number_of_masked_pixels(); ++ i) {
        str << "\n    ";
        ::printFrameCoord(str, frmConfig->masked_pixel_coordinates(i));
      }
    }
    
  }
}

// Method which is called with event data
void 
DumpCamera::event(Event& evt, Env& env)
{

  shared_ptr<Psana::Camera::FrameV1> frmData = evt.get(m_src);
  if (frmData.get()) {
    WithMsgLog(name(), info, str) {
      str << "Camera::FrameV1: width=" << frmData->width()
          << " height=" << frmData->height()
          << " depth=" << frmData->depth()
          << " offset=" << frmData->offset()
          << " data=[" << int(frmData->data()[0])
          << ", " << int(frmData->data()[1])
          << ", " << int(frmData->data()[2]) << ", ...]";
    }
  }

  shared_ptr<Psana::Camera::TwoDGaussianV1> gaussData = evt.get(m_src);
  if (gaussData.get()) {
    WithMsgLog(name(), info, str) {
      str << "Camera::TwoDGaussianV1: integral=" << gaussData->integral()
          << " xmean=" << gaussData->xmean()
          << " ymean=" << gaussData->ymean()
          << " major_axis_width=" << gaussData->major_axis_width()
          << " minor_axis_width=" << gaussData->minor_axis_width()
          << " major_axis_tilt=" << gaussData->major_axis_tilt();
    }
  }

}
  
} // namespace psana_examples
