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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "Exception.h"
#include "types/TypeLib.h"
#include "FrameCoord.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::Camera::FrameFexConfigV1, forwarding)
  FUN0_WRAPPER(pypdsdata::Camera::FrameFexConfigV1, forward_prescale)
  FUN0_WRAPPER(pypdsdata::Camera::FrameFexConfigV1, processing)
  FUN0_WRAPPER(pypdsdata::Camera::FrameFexConfigV1, threshold)
  FUN0_WRAPPER(pypdsdata::Camera::FrameFexConfigV1, number_of_masked_pixels)
  FUN0_WRAPPER(pypdsdata::Camera::FrameFexConfigV1, size)
  PyObject* roiBegin( PyObject* self, PyObject* );
  PyObject* roiEnd( PyObject* self, PyObject* args );
  PyObject* masked_pixel_coordinates( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"forwarding",       forwarding,       METH_NOARGS,  "Returns forwarding policy for frame data." },
    {"forward_prescale", forward_prescale, METH_NOARGS,  "Returns prescale of events with forwarded frames." },
    {"processing",       processing,       METH_NOARGS,  "Returns algorithm to apply to frames to produce processed output." },
    {"roiBegin",         roiBegin,         METH_NOARGS,  "Returns coordinates of start of rectangular region of interest (inclusive)." },
    {"roiEnd",           roiEnd,           METH_NOARGS,  "Returns coordinates of finish of rectangular region of interest (exclusive)." },
    {"threshold",        threshold,        METH_NOARGS,  "Returns pixel data threshold value to apply in processing." },
    {"number_of_masked_pixels", number_of_masked_pixels, METH_NOARGS, "Returns count of masked pixels to exclude from processing." },
    {"masked_pixel_coordinates", masked_pixel_coordinates, METH_NOARGS, "Returns list of masked pixel coordinates." },
    {"size",             size,             METH_NOARGS, "Returns size of this structure (including appended masked pixel coordinates)." },
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

  BaseType::initType( "FrameFexConfigV1", module );
}

namespace {

PyObject*
roiBegin( PyObject* self, PyObject* args)
{
  const Pds::Camera::FrameFexConfigV1* obj = pypdsdata::Camera::FrameFexConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  return pypdsdata::Camera::FrameCoord::PyObject_FromPds ( obj->roiBegin() );
}

PyObject*
roiEnd( PyObject* self, PyObject* args)
{
  const Pds::Camera::FrameFexConfigV1* obj = pypdsdata::Camera::FrameFexConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  return pypdsdata::Camera::FrameCoord::PyObject_FromPds ( obj->roiEnd() );
}

PyObject*
masked_pixel_coordinates( PyObject* self, PyObject* args)
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

}
