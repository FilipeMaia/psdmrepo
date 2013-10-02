//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FramesV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "FramesV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ConfigV1.h"
#include "ConfigV2.h"
#include "FrameV1.h"
#include "../../Exception.h"
#include "../TypeLib.h"
#include "../../pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  PyObject* numLinks( PyObject* self, PyObject* args );
  PyObject* frame( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"numLinks",        numLinks,        METH_VARARGS, 
        "self.numLinks(cfg: ConfigV*) -> int\n\nReturns number of sub-frames" },
    {"frame",        frame,        METH_VARARGS, 
        "self.frame(cfg: ConfigV*, i: int) -> FrameV1\n\nReturns one frame object (:py:class:`FrameV1`) with the specified index." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::PNCCD::FramesV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::PNCCD::FramesV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "FramesV1", module );
}

void
pypdsdata::PNCCD::FramesV1::print(std::ostream& str) const
{
  str << "pnccd.FramesV1(...)";
}

namespace {

PyObject*
numLinks( PyObject* self, PyObject* args )
{
  const Pds::PNCCD::FramesV1* obj = pypdsdata::PNCCD::FramesV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* configObj ;
  if ( not PyArg_ParseTuple( args, "O:PNCCD.FramesV1.numLinks", &configObj ) ) return 0;

  // get Pds::PNCCD::ConfigV1 from argument which could also be of Config2 type
  int numLinks = -1;
  if ( pypdsdata::PNCCD::ConfigV1::Object_TypeCheck( configObj ) ) {
    Pds::PNCCD::ConfigV1* config = pypdsdata::PNCCD::ConfigV1::pdsObject( configObj );
    numLinks = obj->numLinks(*config); 
  } else if ( pypdsdata::PNCCD::ConfigV2::Object_TypeCheck( configObj ) ) {
    Pds::PNCCD::ConfigV2* config = pypdsdata::PNCCD::ConfigV2::pdsObject( configObj );
    numLinks = obj->numLinks(*config); 
  } else {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a PNCCD.ConfigV* object");
    return 0;
  }
  
  using pypdsdata::TypeLib::toPython;
  return toPython(numLinks);
}

PyObject*
frame( PyObject* self, PyObject* args )
{
  const Pds::PNCCD::FramesV1* obj = pypdsdata::PNCCD::FramesV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* configObj ;
  unsigned idx = 0;
  if ( not PyArg_ParseTuple( args, "OI:PNCCD.FramesV1.frame", &configObj, &idx ) ) return 0;

  // get Pds::PNCCD::ConfigV1 from argument which could also be of Config2 type
  const Pds::PNCCD::FrameV1* frame = 0;
  unsigned frameSize = 0;
  if ( pypdsdata::PNCCD::ConfigV1::Object_TypeCheck( configObj ) ) {
    Pds::PNCCD::ConfigV1* config = pypdsdata::PNCCD::ConfigV1::pdsObject( configObj );
    if (idx >= obj->numLinks(*config)) {
      PyErr_SetString(PyExc_IndexError, "index out of range in PNCCD.FramesV1.frame");
      return 0;
    }
    frame = &obj->frame(*config, idx);
    frameSize = obj->_sizeof(*config);
  } else if ( pypdsdata::PNCCD::ConfigV2::Object_TypeCheck( configObj ) ) {
    Pds::PNCCD::ConfigV2* config = pypdsdata::PNCCD::ConfigV2::pdsObject( configObj );
    if (idx >= obj->numLinks(*config)) {
      PyErr_SetString(PyExc_IndexError, "index out of range in PNCCD.FramesV1.frame");
      return 0;
    }
    frame = &obj->frame(*config, idx);
    frameSize = obj->_sizeof(*config);
  } else {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a PNCCD.ConfigV* object");
    return 0;
  }

  return pypdsdata::PNCCD::FrameV1::PyObject_FromPds(const_cast<Pds::PNCCD::FrameV1*>(frame),
      self, frameSize);
}

}
