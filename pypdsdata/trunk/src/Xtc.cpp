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
#include "DetInfo.h"
#include "Exception.h"
#include "ProcInfo.h"
#include "TypeId.h"
#include "XtcIterator.h"
#include "types/TypeLib.h"

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
    { "sizeofPayload",  sizeofPayload,  METH_NOARGS, "Returns the size of payload." },
    { "payload",        Xtc_payload,    METH_NOARGS, "Returns data object. If `contains' is Any returns None. If `contains' is Id_Xtc returns XtcIterator" },
    {0, 0, 0, 0}
   };

  PyGetSetDef getset[] = {
    {"damage",   Xtc_damage,   0, "damage bitmask", 0},
    {"src",      Xtc_src,      0, "data source object, one of BldInfo, DetInfo, or ProcInfo", 0},
    {"contains", Xtc_contains, 0, "TypeId of the contained object(s)", 0},
    {"extent",   extent,       0, "extent size of the XTC", 0},
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
  pypdsdata::Xtc* py_this = (pypdsdata::Xtc*) self;
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  return pypdsdata::Damage::PyObject_FromPds( py_this->m_obj->damage );
}

PyObject*
Xtc_contains( PyObject* self, void* )
{
  pypdsdata::Xtc* py_this = (pypdsdata::Xtc*) self;
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  return pypdsdata::TypeId::PyObject_FromPds( py_this->m_obj->contains );
}

PyObject*
Xtc_src( PyObject* self, void* )
{
  pypdsdata::Xtc* py_this = (pypdsdata::Xtc*) self;
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  const Pds::Src& src = py_this->m_obj->src;
  if ( src.level() == Level::Reporter ) {
    const Pds::BldInfo& info = static_cast<const Pds::BldInfo&>(src);
    return pypdsdata::BldInfo::PyObject_FromPds(info);
  } else if ( src.level() == Level::Source ) {
    const Pds::DetInfo& info = static_cast<const Pds::DetInfo&>(src);
    return pypdsdata::DetInfo::PyObject_FromPds(info);
  } else {
    const Pds::ProcInfo& info = static_cast<const Pds::ProcInfo&>(src);
    return pypdsdata::ProcInfo::PyObject_FromPds(info);
  }
}

PyObject*
Xtc_payload( PyObject* self, PyObject* )
{
  pypdsdata::Xtc* py_this = (pypdsdata::Xtc*) self;
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  if ( py_this->m_obj->contains.id() == Pds::TypeId::Id_Xtc ) {
    return Xtc_iter( self );
  } else if ( py_this->m_obj->contains.id() == Pds::TypeId::Any ) {
    Py_RETURN_NONE;
  } else {
    return pypdsdata::DataObjectFactory::makeObject(*py_this->m_obj, self);
  }
}

}
