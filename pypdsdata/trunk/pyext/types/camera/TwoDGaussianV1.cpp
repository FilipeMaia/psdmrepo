//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Camera_TwoDGaussianV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TwoDGaussianV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::Camera::TwoDGaussianV1, integral)
  FUN0_WRAPPER(pypdsdata::Camera::TwoDGaussianV1, xmean)
  FUN0_WRAPPER(pypdsdata::Camera::TwoDGaussianV1, ymean)
  FUN0_WRAPPER(pypdsdata::Camera::TwoDGaussianV1, major_axis_width)
  FUN0_WRAPPER(pypdsdata::Camera::TwoDGaussianV1, minor_axis_width)
  FUN0_WRAPPER(pypdsdata::Camera::TwoDGaussianV1, major_axis_tilt)
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"integral",         integral,         METH_NOARGS,  "Returns integral statistics as integer number." },
    {"xmean",            xmean,            METH_NOARGS,  "Returns mean X value." },
    {"ymean",            ymean,            METH_NOARGS,  "Returns mean Y value." },
    {"major_axis_width", major_axis_width, METH_NOARGS,  "Returns width of major axis." },
    {"minor_axis_width", minor_axis_width, METH_NOARGS,  "Returns width of minor axis." },
    {"major_axis_tilt",  major_axis_tilt,  METH_NOARGS,  "Returns tilt of major axis." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Camera::TwoDGaussianV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Camera::TwoDGaussianV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "TwoDGaussianV1", module );
}

namespace {
  
PyObject*
_repr( PyObject *self )
{
  Pds::Camera::TwoDGaussianV1* obj = pypdsdata::Camera::TwoDGaussianV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "camera.TwoDGaussianV1(integral=" << obj->integral()
      << ", xmean=" << obj->xmean()
      << ", ymean=" << obj->ymean()
      << ", ...)" ;

  return PyString_FromString( str.str().c_str() );
}

}
