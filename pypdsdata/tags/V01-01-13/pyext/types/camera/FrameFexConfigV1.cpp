//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FrameFexConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "FrameFexConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../EnumType.h"
#include "../../Exception.h"
#include "../TypeLib.h"
#include "FrameCoord.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  pypdsdata::EnumType::Enum fwdEnumValues[] = {
      { "NoFrame", Pds::Camera::FrameFexConfigV1::NoFrame },
      { "FullFrame", Pds::Camera::FrameFexConfigV1::FullFrame },
      { "RegionOfInterest", Pds::Camera::FrameFexConfigV1::RegionOfInterest },
      { 0, 0 }
  };
  pypdsdata::EnumType fwdEnum ( "Forwarding", fwdEnumValues );

  pypdsdata::EnumType::Enum procEnumValues[] = {
      { "NoProcessing", Pds::Camera::FrameFexConfigV1::NoProcessing },
      { "GssFullFrame", Pds::Camera::FrameFexConfigV1::GssFullFrame },
      { "GssRegionOfInterest", Pds::Camera::FrameFexConfigV1::GssRegionOfInterest },
      { "GssThreshold", Pds::Camera::FrameFexConfigV1::GssThreshold },
      { 0, 0 }
  };
  pypdsdata::EnumType procEnum ( "Processing", procEnumValues );


  // methods
  ENUM_FUN0_WRAPPER(pypdsdata::Camera::FrameFexConfigV1, forwarding, fwdEnum)
  FUN0_WRAPPER(pypdsdata::Camera::FrameFexConfigV1, forward_prescale)
  ENUM_FUN0_WRAPPER(pypdsdata::Camera::FrameFexConfigV1, processing, procEnum)
  FUN0_WRAPPER(pypdsdata::Camera::FrameFexConfigV1, threshold)
  FUN0_WRAPPER(pypdsdata::Camera::FrameFexConfigV1, number_of_masked_pixels)
  FUN0_WRAPPER(pypdsdata::Camera::FrameFexConfigV1, roiBegin)
  FUN0_WRAPPER(pypdsdata::Camera::FrameFexConfigV1, roiEnd)
  FUN0_WRAPPER(pypdsdata::Camera::FrameFexConfigV1, masked_pixel_coordinates)

  PyMethodDef methods[] = {
    {"forwarding",       forwarding,       METH_NOARGS,
        "self.forwarding() -> Forwarding enum\n\nReturns forwarding policy for frame data (:py:class:`FrameFexConfigV1.Forwarding`)." },
    {"forward_prescale", forward_prescale, METH_NOARGS,
        "self.forward_prescale() -> int\n\nReturns prescale of events with forwarded frames." },
    {"processing",       processing,       METH_NOARGS,
        "self.processing() -> Processing enum\n\nReturns algorithm to apply to frames to produce processed output (:py:class:`FrameFexConfigV1.Processing`)." },
    {"roiBegin",         roiBegin,         METH_NOARGS,
        "self.roiBegin() -> camera.FrameCoord\n\nReturns coordinates of start of rectangular region of interest (inclusive) as :py:class:`FrameCoord` object." },
    {"roiEnd",           roiEnd,           METH_NOARGS,
        "self.roiEnd() -> camera.FrameCoord\n\nReturns coordinates of finish of rectangular region of interest (exclusive) as :py:class:`FrameCoord` object." },
    {"threshold",        threshold,        METH_NOARGS,
        "self.threshold() -> int\n\nReturns pixel data threshold value to apply in processing." },
    {"number_of_masked_pixels", number_of_masked_pixels, METH_NOARGS,
        "self.number_of_masked_pixels() -> int\n\nReturns count of masked pixels to exclude from processing." },
    {"masked_pixel_coordinates", masked_pixel_coordinates, METH_NOARGS,
        "self.masked_pixel_coordinates() -> list of camera.FrameCoord\n\nReturns list of masked pixel coordinates (:py:class:`FrameCoord` objects)." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Camera::FrameFexConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Camera::FrameFexConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Forwarding", fwdEnum.type() );
  PyDict_SetItemString( tp_dict, "Processing", procEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "FrameFexConfigV1", module );
}

void
pypdsdata::Camera::FrameFexConfigV1::print(std::ostream& str) const
{
  str << "camera.FrameFexConfigV1(forwarding=" << m_obj->forwarding()
      << ", forward_prescale=" << m_obj->forward_prescale()
      << ", processing=" << m_obj->processing()
      << ", roiBegin=(" << m_obj->roiBegin().column() << ',' << m_obj->roiBegin().row() << ')'
      << ", roiEnd=(" << m_obj->roiEnd().column() << ',' << m_obj->roiEnd().row() << ')'
      << ", number_of_masked_pixels=" << m_obj->number_of_masked_pixels()
      << ", ...)" ;
}
