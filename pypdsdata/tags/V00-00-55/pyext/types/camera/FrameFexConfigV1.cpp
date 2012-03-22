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
  FUN0_WRAPPER(pypdsdata::Camera::FrameFexConfigV1, size)
  PyObject* roiBegin( PyObject* self, PyObject* );
  PyObject* roiEnd( PyObject* self, PyObject* );
  PyObject* masked_pixel_coordinates( PyObject* self, PyObject* );
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"forwarding",       forwarding,       METH_NOARGS,
        "self.forwarding() -> Forwarding enum\n\nReturns forwarding policy for frame data." },
    {"forward_prescale", forward_prescale, METH_NOARGS,
        "self.forward_prescale() -> int\n\nReturns prescale of events with forwarded frames." },
    {"processing",       processing,       METH_NOARGS,
        "self.processing() -> Processing enum\n\nReturns algorithm to apply to frames to produce processed output." },
    {"roiBegin",         roiBegin,         METH_NOARGS,
        "self.roiBegin() -> camera.FrameCoord\n\nReturns coordinates of start of rectangular region of interest (inclusive)." },
    {"roiEnd",           roiEnd,           METH_NOARGS,
        "self.roiEnd() -> camera.FrameCoord\n\nReturns coordinates of finish of rectangular region of interest (exclusive)." },
    {"threshold",        threshold,        METH_NOARGS,
        "self.threshold() -> int\n\nReturns pixel data threshold value to apply in processing." },
    {"number_of_masked_pixels", number_of_masked_pixels, METH_NOARGS,
        "self.number_of_masked_pixels() -> int\n\nReturns count of masked pixels to exclude from processing." },
    {"masked_pixel_coordinates", masked_pixel_coordinates, METH_NOARGS,
        "self.masked_pixel_coordinates() -> list of camera.FrameCoord\n\nReturns list of masked pixel coordinates." },
    {"size",             size,             METH_NOARGS,
        "self.size() -> int\n\nReturns size of this structure (including appended masked pixel coordinates)." },
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
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Forwarding", fwdEnum.type() );
  PyDict_SetItemString( tp_dict, "Processing", procEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "FrameFexConfigV1", module );
}

namespace {

PyObject*
roiBegin( PyObject* self, PyObject*)
{
  const Pds::Camera::FrameFexConfigV1* obj = pypdsdata::Camera::FrameFexConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  return pypdsdata::Camera::FrameCoord::PyObject_FromPds ( obj->roiBegin() );
}

PyObject*
roiEnd( PyObject* self, PyObject*)
{
  const Pds::Camera::FrameFexConfigV1* obj = pypdsdata::Camera::FrameFexConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  return pypdsdata::Camera::FrameCoord::PyObject_FromPds ( obj->roiEnd() );
}

PyObject*
masked_pixel_coordinates( PyObject* self, PyObject*)
{
  const Pds::Camera::FrameFexConfigV1* obj = pypdsdata::Camera::FrameFexConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  unsigned size = obj->number_of_masked_pixels() ;
  PyObject* list = PyList_New(size);

  // copy coordinates to the list
  const Pds::Camera::FrameCoord* coords = &(obj->masked_pixel_coordinates());
  for ( unsigned i = 0; i < size; ++ i ) {
    PyObject* obj = pypdsdata::Camera::FrameCoord::PyObject_FromPds(coords[i]) ;
    PyList_SET_ITEM( list, i, obj );
  }

  return list;
}

PyObject*
_repr( PyObject *self )
{
  Pds::Camera::FrameFexConfigV1* obj = pypdsdata::Camera::FrameFexConfigV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "camera.FrameFexConfigV1(forwarding=" << obj->forwarding()
      << ", forward_prescale=" << obj->forward_prescale()
      << ", processing=" << obj->processing()
      << ", roiBegin=(" << obj->roiBegin().column << ',' << obj->roiBegin().row << ')'
      << ", roiEnd=(" << obj->roiEnd().column << ',' << obj->roiEnd().row << ')'
      << ", ...)" ;

  return PyString_FromString( str.str().c_str() );
}

}
