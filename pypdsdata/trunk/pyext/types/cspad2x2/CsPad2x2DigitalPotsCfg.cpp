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
  PyObject* value( PyObject* self, PyObject* args );
  PyObject* pots( PyObject* self, void* );
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"value",     value,      METH_VARARGS, "Returns pot value for a given index." },
    {0, 0, 0, 0}
   };

  PyGetSetDef getset[] = {
    {"pots",         pots,         0, "", 0},
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
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // add an enum analog to this class 
  type->tp_dict = PyDict_New();
  PyObject* val = PyInt_FromLong(Pds::CsPad2x2::PotsPerQuad);
  PyDict_SetItemString( type->tp_dict, "PotsPerQuad", val );
  Py_XDECREF(val);

  BaseType::initType( "CsPad2x2DigitalPotsCfg", module );
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
  
  return PyInt_FromLong( obj->value(index) );
}

PyObject*
pots( PyObject* self, void*)
{
  const Pds::CsPad2x2::CsPad2x2DigitalPotsCfg* obj = pypdsdata::CsPad2x2::CsPad2x2DigitalPotsCfg::pdsObject( self );
  if ( not obj ) return 0;

  PyObject* list = PyList_New( Pds::CsPad2x2::PotsPerQuad );
  for ( unsigned i = 0 ; i < Pds::CsPad2x2::PotsPerQuad ; ++ i ) {
    PyList_SET_ITEM( list, i, PyInt_FromLong(obj->pots[i]) );
  }

  return list;
}

PyObject*
_repr( PyObject *self )
{
  const Pds::CsPad2x2::CsPad2x2DigitalPotsCfg* pdsObj = pypdsdata::CsPad2x2::CsPad2x2DigitalPotsCfg::pdsObject( self );
  if(not pdsObj) return 0;

  std::ostringstream str;
  str << "cspad2x2.CsPad2x2DigitalPotsCfg([" << int(pdsObj->pots[0])
      << ", " << int(pdsObj->pots[1])
      << ", " << int(pdsObj->pots[2])
      << ", " << int(pdsObj->pots[3])
      << ", ...])";
  return PyString_FromString( str.str().c_str() );
}

}

