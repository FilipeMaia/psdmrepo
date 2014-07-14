//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_DataDescV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataDescV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"
#include "ConfigV1.h"
#include "DataDescV1Elem.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  PyObject* data( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"data",        data,         METH_VARARGS,
        "self.data(cfg: ConfigV1) -> list of DataDescV1Elem\n\nReturns list of :py:class:`DataDescV1Elem` objects, "
        "one object per channel." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Acqiris::DataDescV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Acqiris::DataDescV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "DataDescV1", module );
}

void 
pypdsdata::Acqiris::DataDescV1::print(std::ostream& out) const
{
  out << "acqiris.DataDescV1(...)";
}

namespace {

PyObject*
data( PyObject* self, PyObject* args )
{
  Pds::Acqiris::DataDescV1* obj = pypdsdata::Acqiris::DataDescV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* pyconfig ;
  if ( not PyArg_ParseTuple( args, "O:Acqiris.DataDescV1.data", &pyconfig ) ) return 0;

  // check type
  if ( not pypdsdata::Acqiris::ConfigV1::Object_TypeCheck( pyconfig ) ) {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a Acqiris.ConfigV1 object");
    return 0;
  }

  // convert to Pds config object
  const Pds::Acqiris::ConfigV1* config = pypdsdata::Acqiris::ConfigV1::pdsObject( pyconfig );

  const unsigned nchan = config->nbrChannels();
  PyObject* list = PyList_New(nchan);
  for (unsigned i = 0; i != nchan; ++ i) {
    const Pds::Acqiris::DataDescV1Elem& elem = obj->data(*config, i);
    PyObject* pyelem = pypdsdata::Acqiris::DataDescV1Elem::PyObject_FromPds(const_cast<Pds::Acqiris::DataDescV1Elem*>(&elem),
        self, elem._sizeof(*config));
    PyList_SET_ITEM( list, i, pyelem );
  }

  return list;
}

}

