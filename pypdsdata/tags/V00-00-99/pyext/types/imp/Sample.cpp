//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Sample...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "Sample.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../EnumType.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // list of enums
  pypdsdata::TypeLib::EnumEntry enums[] = {
        { "channelsPerDevice",      Pds::Imp::channelsPerDevice },
        { 0, 0 }
  };

  // type-specific methods
  PyObject* channels( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    { "channels",       channels,       METH_NOARGS,
        "self.channels() -> list of int\n\nReturns list of integer numbers, one value per channel." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Imp::Sample class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Imp::Sample::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  pypdsdata::TypeLib::DefineEnums( type->tp_dict, ::enums );

  BaseType::initType( "Sample", module );
}

void
pypdsdata::Imp::Sample::print(std::ostream& out) const
{
  out << "imp.Sample([";
  for (unsigned i = 0; i != Pds::Imp::channelsPerDevice; ++ i) {
    if (i > 0) out << ", ";
    out << const_cast<Pds::Imp::Sample&>(m_obj).channel(i);
  }
  out << "])";
}

namespace {

PyObject*
channels( PyObject* self, PyObject* args )
{
  Pds::Imp::Sample& obj = pypdsdata::Imp::Sample::pdsObject( self );

  unsigned size = Pds::Imp::channelsPerDevice;
  PyObject* list = PyList_New(size);
  // copy values to the list
  for ( unsigned i = 0; i < size; ++ i ) {
    PyList_SET_ITEM(list, i, PyInt_FromLong(obj.channel(i)));
  }

  return list;
}

}
