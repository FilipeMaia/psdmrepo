//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadDigitalPotsCfg...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CsPadDigitalPotsCfg.h"

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

  PyMethodDef methods[] = {
    {"value",     value,      METH_VARARGS, "self.value(i: int) -> int\n\nReturns pot value for a given index." },
    {0, 0, 0, 0}
   };

  PyGetSetDef getset[] = {
    {"pots",         pots,         0, "List of PotsPerQuad integers", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::CsPad::CsPadDigitalPotsCfg class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::CsPad::CsPadDigitalPotsCfg::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_getset = ::getset;

  // add an enum analog to this class 
  type->tp_dict = PyDict_New();
  PyObject* val = PyInt_FromLong(Pds::CsPad::PotsPerQuad);
  PyDict_SetItemString( type->tp_dict, "PotsPerQuad", val );
  Py_XDECREF(val);

  BaseType::initType( "CsPadDigitalPotsCfg", module );
}

namespace {

PyObject*
value( PyObject* self, PyObject* args )
{
  const Pds::CsPad::CsPadDigitalPotsCfg* obj = pypdsdata::CsPad::CsPadDigitalPotsCfg::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned index ;
  if ( not PyArg_ParseTuple( args, "I:CsPadDigitalPotsCfg_value", &index ) ) return 0;

  if ( index >= Pds::CsPad::PotsPerQuad ) {
    PyErr_SetString(PyExc_IndexError, "index outside of range [0..PotsPerQuad) in CsPadDigitalPotsCfg.value()");
    return 0;
  }
  
  return PyInt_FromLong( obj->value(index) );
}

PyObject*
pots( PyObject* self, void*)
{
  const Pds::CsPad::CsPadDigitalPotsCfg* obj = pypdsdata::CsPad::CsPadDigitalPotsCfg::pdsObject( self );
  if ( not obj ) return 0;

  PyObject* list = PyList_New( Pds::CsPad::PotsPerQuad );
  for ( unsigned i = 0 ; i < Pds::CsPad::PotsPerQuad ; ++ i ) {
    PyList_SET_ITEM( list, i, PyInt_FromLong(obj->pots[i]) );
  }

  return list;
}


}

