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

  // standard Python stuff
  PyObject* _repr( PyObject *self );

  // methods
  MEMBER_WRAPPER(pypdsdata::EpicsPvTime, iPvId)
  MEMBER_WRAPPER(pypdsdata::EpicsPvTime, iDbrType)
  MEMBER_WRAPPER(pypdsdata::EpicsPvTime, iNumElements)
  PyObject* EpicsPvTime_status( PyObject* self, void* );
  PyObject* EpicsPvTime_severity( PyObject* self, void* );
  PyObject* EpicsPvTime_stamp( PyObject* self, void* );
  PyObject* EpicsPvTime_value( PyObject* self, void* );
  PyObject* EpicsPvTime_values( PyObject* self, void* );
  PyObject* EpicsPvTime_getnewargs( PyObject* self, PyObject* );

  PyGetSetDef getset[] = {
    {"iPvId",        iPvId,                0, "Pv Id", 0},
    {"iDbrType",     iDbrType,             0, "Epics Data Type", 0},
    {"iNumElements", iNumElements,         0, "Size of Pv Array", 0},
    {"status",       EpicsPvTime_status,   0, "Status value", 0},
    {"severity",     EpicsPvTime_severity, 0, "Severity value", 0},
    {"stamp",        EpicsPvTime_stamp,    0, "EPICS timestamp value of type epicsTimeStamp", 0},
    {"value",        EpicsPvTime_value,    0, "PV value, always a single value, for arrays it is first element", 0},
    {"values",       EpicsPvTime_values,   0, "List of PV values of size [iNumElements]", 0},
    {0, 0, 0, 0, 0}
  };

  PyMethodDef methods[] = {
    { "__getnewargs__",    EpicsPvTime_getnewargs, METH_NOARGS, "Pickle support" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EpicsPvTime class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::EpicsPvTime::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_getset = ::getset;
  type->tp_str = _repr ;
  type->tp_repr = _repr ;

  BaseType::initType( "EpicsPvTime", module );
}


namespace {


template <int iDbrType>
inline
void
print(std::ostream& out, const Pds::EpicsPvHeader& header)
{
  typedef Pds::EpicsPvTime<iDbrType> PVType;
  const PVType& obj = static_cast<const PVType&>(header);
  out << "id=" << obj.iPvId
      << ", status=" << obj.status
      << ", severity=" << obj.severity
      << ", value=" << obj.value;
}


PyObject*
_repr( PyObject *self )
{
  Pds::EpicsPvHeader* obj = pypdsdata::EpicsPvTime::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "EpicsPvTime(";
  
  
  switch ( obj->iDbrType ) {

  case DBR_TIME_STRING:
    print<DBR_STRING>(str, *obj);
    break;

  case DBR_TIME_SHORT:
    print<DBR_SHORT>(str, *obj);
    break;

  case DBR_TIME_FLOAT:
    print<DBR_FLOAT>(str, *obj);
    break;

  case DBR_TIME_ENUM:
    print<DBR_ENUM>(str, *obj);
    break;

  case DBR_TIME_CHAR:
    print<DBR_CHAR>(str, *obj);
    break;

  case DBR_TIME_LONG:
    print<DBR_LONG>(str, *obj);
    break;

  case DBR_TIME_DOUBLE:
    print<DBR_DOUBLE>(str, *obj);
    break;

  default:
    str << "id=" << obj->iPvId;
  }

  str << ")";
  
  return PyString_FromString( str.str().c_str() );
}

PyObject*
EpicsPvTime_status( PyObject* self, void* )
{
  pypdsdata::EpicsPvTime* py_this = static_cast<pypdsdata::EpicsPvTime*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  // all DBR_TIME_* types share the same layout, cast to arbitrary type
  typedef Pds::EpicsPvTime<DBR_SHORT> TimeType ;
  TimeType* pvTime = (TimeType*)py_this->m_obj;

  return pypdsdata::TypeLib::toPython( pvTime->status );
}

PyObject*
EpicsPvTime_severity( PyObject* self, void* )
{
  pypdsdata::EpicsPvTime* py_this = static_cast<pypdsdata::EpicsPvTime*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  // all DBR_TIME_* types share the same layout, cast to arbitrary type
  typedef Pds::EpicsPvTime<DBR_SHORT> TimeType ;
  TimeType* pvTime = (TimeType*)py_this->m_obj;

  return pypdsdata::TypeLib::toPython( pvTime->severity );
}

PyObject*
EpicsPvTime_stamp( PyObject* self, void* )
{
  pypdsdata::EpicsPvTime* py_this = static_cast<pypdsdata::EpicsPvTime*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  // all DBR_TIME_* types share the same layout, cast to arbitrary type
  typedef Pds::EpicsPvTime<DBR_SHORT> TimeType ;
  TimeType* pvTime = (TimeType*)py_this->m_obj;

  return pypdsdata::Epics::epicsTimeStamp::PyObject_FromPds( pvTime->stamp );
}

template <int iDbrType>
inline
PyObject*
getValue( Pds::EpicsPvHeader* header, int index = 0 )
{
  typedef Pds::EpicsPvTime<iDbrType> TimeType ;
  TimeType* obj = static_cast<TimeType*>(header);
  return pypdsdata::TypeLib::toPython( (&obj->value)[index] );
}

PyObject*
EpicsPvTime_value( PyObject* self, void* )
{
  pypdsdata::EpicsPvTime* py_this = static_cast<pypdsdata::EpicsPvTime*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  if ( py_this->m_obj->iNumElements <= 0 ) {
    PyErr_SetString(PyExc_TypeError, "Non-positive PV array size");
    return 0;
  }

  PyObject* obj = 0;
  switch ( py_this->m_obj->iDbrType ) {

  case DBR_TIME_STRING:
    obj = getValue<DBR_STRING>( py_this->m_obj );
    break;

  case DBR_TIME_SHORT:
    obj = getValue<DBR_SHORT>( py_this->m_obj );
    break;

  case DBR_TIME_FLOAT:
    obj = getValue<DBR_FLOAT>( py_this->m_obj );
    break;

  case DBR_TIME_ENUM:
    obj = getValue<DBR_ENUM>( py_this->m_obj );
    break;

  case DBR_TIME_CHAR:
    obj = getValue<DBR_CHAR>( py_this->m_obj );
    break;

  case DBR_TIME_LONG:
    obj = getValue<DBR_LONG>( py_this->m_obj );
    break;

  case DBR_TIME_DOUBLE:
    obj = getValue<DBR_DOUBLE>( py_this->m_obj );
    break;

  default:
    PyErr_SetString(PyExc_TypeError, "Unexpected PV type");
  }
  return obj;
}

PyObject*
EpicsPvTime_values( PyObject* self, void* )
{
  pypdsdata::EpicsPvTime* py_this = static_cast<pypdsdata::EpicsPvTime*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  int size = py_this->m_obj->iNumElements;
  if ( size < 0 ) {
    PyErr_SetString(PyExc_TypeError, "Negative PV array size");
    return 0;
  }

  PyObject* list = PyList_New( size );
  switch ( py_this->m_obj->iDbrType ) {

  case DBR_TIME_STRING:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<DBR_STRING>( py_this->m_obj ) );
    }
    break;

  case DBR_TIME_SHORT:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<DBR_SHORT>( py_this->m_obj ) );
    }
    break;

  case DBR_TIME_FLOAT:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<DBR_FLOAT>( py_this->m_obj ) );
    }
    break;

  case DBR_TIME_ENUM:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<DBR_ENUM>( py_this->m_obj ) );
    }
    break;

  case DBR_TIME_CHAR:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<DBR_CHAR>( py_this->m_obj ) );
    }
    break;

  case DBR_TIME_LONG:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<DBR_LONG>( py_this->m_obj ) );
    }
    break;

  case DBR_TIME_DOUBLE:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<DBR_DOUBLE>( py_this->m_obj ) );
    }
    break;

  default:
    PyErr_SetString(PyExc_TypeError, "Unexpected PV type");
  }
  return list;
}

PyObject*
EpicsPvTime_getnewargs( PyObject* self, PyObject* )
{
  pypdsdata::EpicsPvTime* py_this = static_cast<pypdsdata::EpicsPvTime*>(self);
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
