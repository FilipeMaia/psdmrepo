//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpicsPvTime...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "EpicsPvTime.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "epicsTimeStamp.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  template <typename EpicsType>
  inline
  void
  print(std::ostream& out, const Pds::Epics::EpicsPvHeader& header)
  {
    const EpicsType& obj = static_cast<const EpicsType&>(header);
    out << "id=" << obj.pvId()
        << ", status=" << obj.status()
        << ", severity=" << obj.severity()
        << ", value=" << obj.value(0);
  }

  namespace gs {
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Epics::EpicsPvTime, pvId)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Epics::EpicsPvTime, dbrType)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Epics::EpicsPvTime, numElements)
  }

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"iPvId",        gs::pvId,                 0, "Integer number, Pv Id", 0},
    {"iDbrType",     gs::dbrType,              0, "Integer number, Epics Data Type", 0},
    {"iNumElements", gs::numElements,          0, "Integer number, Size of Pv Array", 0},
    {0, 0, 0, 0, 0}
  };

  namespace mm {
  FUN0_WRAPPER(pypdsdata::Epics::EpicsPvTime, pvId)
  FUN0_WRAPPER(pypdsdata::Epics::EpicsPvTime, dbrType)
  FUN0_WRAPPER(pypdsdata::Epics::EpicsPvTime, numElements)
  PyObject* EpicsPvTime_status( PyObject* self, PyObject* );
  PyObject* EpicsPvTime_severity( PyObject* self, PyObject* );
  PyObject* EpicsPvTime_stamp( PyObject* self, PyObject* );
  PyObject* EpicsPvTime_value( PyObject* self, PyObject* );
  PyObject* EpicsPvTime_values( PyObject* self, PyObject* );
  PyObject* EpicsPvTime_getnewargs( PyObject* self, PyObject* );
  }
  
  PyMethodDef methods[] = {
    {"status",          mm::EpicsPvTime_status,         METH_NOARGS, "Integer number, Status value"},
    {"severity",        mm::EpicsPvTime_severity,       METH_NOARGS, "Integer number, Severity value"},
    {"stamp",           mm::EpicsPvTime_stamp,          METH_NOARGS, "EPICS timestamp value of type epicsTimeStamp"},
    {"value",           mm::EpicsPvTime_value,          METH_NOARGS, "PV value (number or string), always a single value, for arrays it is first element"},
    {"values",          mm::EpicsPvTime_values,         METH_NOARGS, "List of PV values of size [iNumElements]"},
    { "__getnewargs__", mm::EpicsPvTime_getnewargs,     METH_NOARGS, "Pickle support" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EpicsPvTime class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Epics::EpicsPvTime::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_getset = ::getset;

  BaseType::initType( "EpicsPvTime", module );
}

void
pypdsdata::Epics::EpicsPvTime::print(std::ostream& str) const
{
  str << "EpicsPvTime(";

  switch ( m_obj->dbrType() ) {

  case Pds::Epics::DBR_TIME_STRING:
    ::print<Pds::Epics::EpicsPvTimeString>(str, *m_obj);
    break;

  case Pds::Epics::DBR_TIME_SHORT:
    ::print<Pds::Epics::EpicsPvTimeShort>(str, *m_obj);
    break;

  case Pds::Epics::DBR_TIME_FLOAT:
    ::print<Pds::Epics::EpicsPvTimeFloat>(str, *m_obj);
    break;

  case Pds::Epics::DBR_TIME_ENUM:
    ::print<Pds::Epics::EpicsPvTimeEnum>(str, *m_obj);
    break;

  case Pds::Epics::DBR_TIME_CHAR:
    ::print<Pds::Epics::EpicsPvTimeChar>(str, *m_obj);
    break;

  case Pds::Epics::DBR_TIME_LONG:
    ::print<Pds::Epics::EpicsPvTimeLong>(str, *m_obj);
    break;

  case Pds::Epics::DBR_TIME_DOUBLE:
    ::print<Pds::Epics::EpicsPvTimeDouble>(str, *m_obj);
    break;

  default:
    str << "id=" << m_obj->pvId();
  }

  str << ")";
}

namespace {

PyObject*
EpicsPvTime_status( PyObject* self, void* )
{
  const Pds::Epics::EpicsPvTimeHeader* obj = pypdsdata::Epics::EpicsPvTime::pdsObject( self );
  if ( not obj ) return 0;

  // all DBR_TIME_* types share the same layout, cast to arbitrary type
  typedef Pds::Epics::EpicsPvTimeShort TimeType ;
  const TimeType* pvTime = static_cast<const TimeType*>(obj);

  return pypdsdata::TypeLib::toPython( pvTime->status() );
}

PyObject*
EpicsPvTime_severity( PyObject* self, void* )
{
  const Pds::Epics::EpicsPvTimeHeader* obj = pypdsdata::Epics::EpicsPvTime::pdsObject( self );
  if ( not obj ) return 0;

  // all DBR_TIME_* types share the same layout, cast to arbitrary type
  typedef Pds::Epics::EpicsPvTimeShort TimeType ;
  const TimeType* pvTime = static_cast<const TimeType*>(obj);

  return pypdsdata::TypeLib::toPython( pvTime->severity() );
}

PyObject*
EpicsPvTime_stamp( PyObject* self, void* )
{
  const Pds::Epics::EpicsPvTimeHeader* obj = pypdsdata::Epics::EpicsPvTime::pdsObject( self );
  if ( not obj ) return 0;

  // all DBR_TIME_* types share the same layout, cast to arbitrary type
  typedef Pds::Epics::EpicsPvTimeShort TimeType ;
  const TimeType* pvTime = static_cast<const TimeType*>(obj);

  return pypdsdata::Epics::epicsTimeStamp::PyObject_FromPds( pvTime->stamp() );
}

// return PV data, if index is -1 the return an array, if index >= 0 return one element
template <typename EpicsType>
inline
PyObject*
getValue( const Pds::Epics::EpicsPvTimeHeader& header, int index = -1 )
{
  const EpicsType& obj = static_cast<const EpicsType&>(header);
  using pypdsdata::TypeLib::toPython;
  if (index < 0) {
    const unsigned size = obj.numElements();
    PyObject* list = PyList_New(size);
    for ( unsigned i = 0; i < size; ++ i ) {
      using pypdsdata::TypeLib::toPython;
      PyList_SET_ITEM(list, i, toPython(obj.value(i)));
    }
    return list;
  } else {
    return toPython(obj.value(index));
  }
}

PyObject*
EpicsPvTime_value( PyObject* self, void* )
{
  const Pds::Epics::EpicsPvTimeHeader* obj = pypdsdata::Epics::EpicsPvTime::pdsObject( self );
  if ( not obj ) return 0;

  if ( obj->numElements() <= 0 ) {
    PyErr_SetString(PyExc_TypeError, "Non-positive PV array size");
    return 0;
  }

  PyObject* pyobj = 0;
  switch ( obj->dbrType() ) {

  case Pds::Epics::DBR_TIME_STRING:
    pyobj = getValue<Pds::Epics::EpicsPvTimeString>( *obj, 0 );
    break;

  case Pds::Epics::DBR_TIME_SHORT:
    pyobj = getValue<Pds::Epics::EpicsPvTimeShort>( *obj, 0 );
    break;

  case Pds::Epics::DBR_TIME_FLOAT:
    pyobj = getValue<Pds::Epics::EpicsPvTimeFloat>( *obj, 0 );
    break;

  case Pds::Epics::DBR_TIME_ENUM:
    pyobj = getValue<Pds::Epics::EpicsPvTimeEnum>( *obj, 0 );
    break;

  case Pds::Epics::DBR_TIME_CHAR:
    pyobj = getValue<Pds::Epics::EpicsPvTimeChar>( *obj, 0 );
    break;

  case Pds::Epics::DBR_TIME_LONG:
    pyobj = getValue<Pds::Epics::EpicsPvTimeLong>( *obj, 0 );
    break;

  case Pds::Epics::DBR_TIME_DOUBLE:
    pyobj = getValue<Pds::Epics::EpicsPvTimeDouble>( *obj, 0 );
    break;

  default:
    PyErr_SetString(PyExc_TypeError, "Unexpected PV type");
  }
  return pyobj;
}

PyObject*
EpicsPvTime_values( PyObject* self, void* )
{
  const Pds::Epics::EpicsPvTimeHeader* obj = pypdsdata::Epics::EpicsPvTime::pdsObject( self );
  if ( not obj ) return 0;

  int size = obj->numElements();
  if ( size < 0 ) {
    PyErr_SetString(PyExc_TypeError, "Negative PV array size");
    return 0;
  }

  PyObject* list = 0;
  switch ( obj->dbrType() ) {

  case Pds::Epics::DBR_TIME_STRING:
    list = getValue<Pds::Epics::EpicsPvTimeString>( *obj );
    break;

  case Pds::Epics::DBR_TIME_SHORT:
    list = getValue<Pds::Epics::EpicsPvTimeShort>( *obj );
    break;

  case Pds::Epics::DBR_TIME_FLOAT:
    list = getValue<Pds::Epics::EpicsPvTimeFloat>( *obj );
    break;

  case Pds::Epics::DBR_TIME_ENUM:
    list = getValue<Pds::Epics::EpicsPvTimeEnum>( *obj );
    break;

  case Pds::Epics::DBR_TIME_CHAR:
    list = getValue<Pds::Epics::EpicsPvTimeChar>( *obj );
    break;

  case Pds::Epics::DBR_TIME_LONG:
    list = getValue<Pds::Epics::EpicsPvTimeLong>( *obj );
    break;

  case Pds::Epics::DBR_TIME_DOUBLE:
    list = getValue<Pds::Epics::EpicsPvTimeDouble>( *obj );
    break;

  default:
    PyErr_SetString(PyExc_TypeError, "Unexpected PV type");
  }
  return list;
}

PyObject*
EpicsPvTime_getnewargs( PyObject* self, PyObject* )
{
  pypdsdata::Epics::EpicsPvTime* py_this = static_cast<pypdsdata::Epics::EpicsPvTime*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  size_t size = py_this->m_size;
  const char* data = (const char*)py_this->m_obj;
  PyObject* pydata = PyString_FromStringAndSize(data, size);

  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, pydata);

  return args;
}

}
