//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"
#include "../camera/FrameCoord.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::PNCCD::ConfigV1, numLinks)
  FUN0_WRAPPER(pypdsdata::PNCCD::ConfigV1, payloadSizePerLink)
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"numLinks",           numLinks,           METH_NOARGS,  "self.numLinks() -> int\n\nReturns number of links." },
    {"payloadSizePerLink", payloadSizePerLink, METH_NOARGS,  "self.payloadSizePerLink() -> int\n\nReturns data size per link." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::PNCCD::ConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::PNCCD::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "ConfigV1", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::PNCCD::ConfigV1* obj = pypdsdata::PNCCD::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  char buf[96];
  snprintf( buf, sizeof buf, "pnccd.ConfigV1(numLinks=%d, payloadSizePerLink=%d)",
            obj->numLinks(), obj->payloadSizePerLink() );
  return PyString_FromString( buf );
}

}
