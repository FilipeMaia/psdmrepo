//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Xtc...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "Xtc.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "BldInfo.h"
#include "Damage.h"
#include "DataObjectFactory.h"
#include "Exception.h"
#include "Src.h"
#include "TypeId.h"
#include "XtcIterator.h"
#include "XtcEmbedded.h"
#include "types/TypeLib.h"

#include "pdsdata/compress/CompressedXtc.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  PyObject* Xtc_iter( PyObject* self );

  // type-specific methods
  PyObject* Xtc_damage( PyObject* self, void* );
  PyObject* Xtc_contains( PyObject* self, void* );
  PyObject* Xtc_src( PyObject* self, void* );
  MEMBER_WRAPPER(pypdsdata::Xtc, extent);
  FUN0_WRAPPER(pypdsdata::Xtc, sizeofPayload);
  PyObject* Xtc_payload( PyObject* self, PyObject* );

  PyMethodDef methods[] = {
    { "sizeofPayload",  sizeofPayload,  METH_NOARGS, "self.sizeofPayload() -> int\n\nReturns the size of payload." },
    { "payload",        Xtc_payload,    METH_NOARGS, 
        "self.payload() -> object\n\nReturns data object. If 'contains' is Any returns None. If 'contains' is Id_Xtc returns :py:class:`XtcIterator`" },
    {0, 0, 0, 0}
   };

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"damage",   Xtc_damage,   0, "Attribute which contains damage bitmask (xtc.Damage)", 0},
    {"src",      Xtc_src,      0, "Attribute with data source object, one of :py:class:`BldInfo`, :py:class:`DetInfo`, or :py:class:`ProcInfo`", 0},
    {"contains", Xtc_contains, 0, "Attribute containing TypeId of the contained object(s)", 0},
    {"extent",   extent,       0, "Attribute with extent size of the XTC", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Xtc class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------


namespace pypdsdata {

void
Xtc::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_getset = ::getset;
  type->tp_iter = Xtc_iter;

  BaseType::initType( "Xtc", module );
}

// Check object type
bool
Xtc::Xtc_Check( PyObject* obj )
{
  return PyObject_TypeCheck( obj, BaseType::typeObject() );
}

// REturns a pointer to Pds object
Pds::Xtc*
Xtc::Xtc_AsPds( PyObject* obj )
{
  if ( obj->ob_type == BaseType::typeObject() ) {
    return ((pypdsdata::Xtc*)obj)->m_obj;
  } else {
    return 0;
  }
}

} // namespace pypdsdata


namespace {

PyObject*
Xtc_iter( PyObject* self )
{
  pypdsdata::Xtc* py_this = (pypdsdata::Xtc*) self;
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  // it will throw exception if this xtc of incorrect type
  return pypdsdata::XtcIterator::XtcIterator_FromXtc( py_this->m_obj, self );
}

PyObject*
Xtc_damage( PyObject* self, void* )
{
  pypdsdata::Xtc* py_this = static_cast<pypdsdata::Xtc*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  return pypdsdata::Damage::PyObject_FromPds( py_this->m_obj->damage );
}

PyObject*
Xtc_contains( PyObject* self, void* )
{
  pypdsdata::Xtc* py_this = static_cast<pypdsdata::Xtc*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  return pypdsdata::TypeId::PyObject_FromPds( py_this->m_obj->contains );
}

PyObject*
Xtc_src( PyObject* self, void* )
{
  pypdsdata::Xtc* py_this = static_cast<pypdsdata::Xtc*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  return toPython(py_this->m_obj->src);
}

/**
 *  Return payload as Python object. May return None in some cases 
 *  (for Any object or non-fatal errors such as zero-payload Epics) or 
 *  throw an exception if error is fatal.
 */
PyObject*
Xtc_payload( PyObject* self, PyObject* )
{
  pypdsdata::Xtc* py_this = static_cast<pypdsdata::Xtc*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  if ( py_this->m_obj->contains.id() == Pds::TypeId::Id_Xtc ) {
    return Xtc_iter( self );
  } else if ( py_this->m_obj->contains.id() == Pds::TypeId::Any ) {
    Py_RETURN_NONE;
  } else {
    if (py_this->m_obj->contains.compressed()) {
      boost::shared_ptr<Pds::Xtc> xtc = Pds::CompressedXtc::uncompress(*py_this->m_obj);
      if (!xtc) {
        PyErr_SetString(pypdsdata::exceptionType(), "Error: XTC decompression has failed");
        return 0;
      }
      PyObject* parent = pypdsdata::XtcEmbedded::PyObject_FromPds(xtc);
      PyObject* obj = pypdsdata::DataObjectFactory::makeObject(*xtc.get(), parent);
      Py_CLEAR(parent);
      return obj;
    }
    return pypdsdata::DataObjectFactory::makeObject(*py_this->m_obj, self);
  }
}

}
