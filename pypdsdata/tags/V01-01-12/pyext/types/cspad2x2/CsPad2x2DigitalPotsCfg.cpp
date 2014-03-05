//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2DigitalPotsCfg...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CsPad2x2DigitalPotsCfg.h"

//-----------------
// C/C++ Headers --
//-----------------

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
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::CsPad2x2::CsPad2x2DigitalPotsCfg, pots)
  PyObject* value( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"value",     value,      METH_VARARGS, "self.value(i: int) -> int\n\nReturns pot value for a given index." },
    {0, 0, 0, 0}
   };

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"pots",         pots,         0, "List of PotsPerQuad integers", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::CsPad2x2::CsPad2x2DigitalPotsCfg class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::CsPad2x2::CsPad2x2DigitalPotsCfg::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_getset = ::getset;

  // add an enum analog to this class 
  type->tp_dict = PyDict_New();
  PyObject* val = PyInt_FromLong(Pds::CsPad2x2::PotsPerQuad);
  PyDict_SetItemString( type->tp_dict, "PotsPerQuad", val );
  Py_XDECREF(val);

  BaseType::initType( "CsPad2x2DigitalPotsCfg", module );
}

void
pypdsdata::CsPad2x2::CsPad2x2DigitalPotsCfg::print(std::ostream& str) const
{
  const ndarray<const uint8_t, 1>& pots = m_obj->pots();
  str << "cspad2x2.CsPad2x2DigitalPotsCfg([" << int(pots[0])
      << ", " << int(pots[1])
      << ", " << int(pots[2])
      << ", " << int(pots[3])
      << ", ...])";
}

namespace {

PyObject*
value( PyObject* self, PyObject* args )
{
  const Pds::CsPad2x2::CsPad2x2DigitalPotsCfg* obj = pypdsdata::CsPad2x2::CsPad2x2DigitalPotsCfg::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned index ;
  if ( not PyArg_ParseTuple( args, "I:CsPad2x2DigitalPotsCfg_value", &index ) ) return 0;

  if ( index >= Pds::CsPad2x2::PotsPerQuad ) {
    PyErr_SetString(PyExc_IndexError, "index outside of range [0..PotsPerQuad) in CsPad2x2DigitalPotsCfg.value()");
    return 0;
  }
  
  return PyInt_FromLong( obj->pots()[index] );
}

}

