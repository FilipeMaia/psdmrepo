//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataV2.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ConfigV2.h"
#include "ConfigV3.h"
#include "ConfigV4.h"
#include "ConfigV5.h"
#include "ElementV2.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  PyObject* quads( PyObject* self, PyObject* args );
  template <typename Config> 
  PyObject* quads( PyObject* self, const Config& config, unsigned iq );

  PyMethodDef methods[] = {
    {"quads",       quads,       METH_VARARGS,  
        "self.quads(cfg: ConfigV*, iq: int) -> ElementV2\n\nReturns quadrant data (:py:class:`ElementV2`) for specified quadrant index" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::CsPad::DataV2 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::CsPad::DataV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "DataV2", module );
}

namespace {

PyObject*
quads( PyObject* self, PyObject* args )
{
  // parse args
  PyObject* configObj ;
  unsigned iq;
  if ( not PyArg_ParseTuple( args, "OI:cspad.DataV2.data", &configObj, &iq ) ) return 0;

  if ( pypdsdata::CsPad::ConfigV2::Object_TypeCheck( configObj ) ) {
    const Pds::CsPad::ConfigV2* config = pypdsdata::CsPad::ConfigV2::pdsObject( configObj );
    return quads(self, *config, iq);
  } else if ( pypdsdata::CsPad::ConfigV3::Object_TypeCheck( configObj ) ) {
    const Pds::CsPad::ConfigV3* config = pypdsdata::CsPad::ConfigV3::pdsObject( configObj );
    return quads(self, *config, iq);
  } else if ( pypdsdata::CsPad::ConfigV4::Object_TypeCheck( configObj ) ) {
    const Pds::CsPad::ConfigV4* config = pypdsdata::CsPad::ConfigV4::pdsObject( configObj );
    return quads(self, *config, iq);
  } else if ( pypdsdata::CsPad::ConfigV5::Object_TypeCheck( configObj ) ) {
    const Pds::CsPad::ConfigV5* config = pypdsdata::CsPad::ConfigV5::pdsObject( configObj );
    return quads(self, *config, iq);
  } else {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a cspad.ConfigV* object");
    return 0;
  }
}

template <typename Config> 
PyObject* quads( PyObject* self, const Config& config, unsigned iq )
{
  Pds::CsPad::DataV2* obj = pypdsdata::CsPad::DataV2::pdsObject( self );
  if ( not obj ) return 0;

  // check quad index
  if (iq >= config.numQuads()) {
    PyErr_SetString(PyExc_IndexError, "quadrant index outside of range in cspad.DataV2.quads()");
    return 0;
  }

  // get quad data
  const Pds::CsPad::ElementV2& quad = obj->quads(config, iq);
  
  // convert to Python
  return pypdsdata::CsPad::ElementV2::PyObject_FromPds( const_cast<Pds::CsPad::ElementV2*>(&quad), 
      self, quad._sizeof(config));
}

}
