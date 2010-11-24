//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpicsPvCtrl...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "EpicsPvCtrl.h"

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

  template <int DBR_TYPE>
  struct LimitUnitGetter {

    static PyObject* get( Pds::EpicsPvCtrlHeader* header, void* closure ) {

      Pds::EpicsPvCtrl<DBR_TYPE>* ctrl = (Pds::EpicsPvCtrl<DBR_TYPE>*)header ;
      switch( *(char*)closure ) {
      case 'u' :
        return pypdsdata::TypeLib::toPython( ctrl->units );
      case 'd' :
        return pypdsdata::TypeLib::toPython( ctrl->lower_disp_limit );
      case 'D' :
        return pypdsdata::TypeLib::toPython( ctrl->upper_disp_limit );
      case 'a' :
        return pypdsdata::TypeLib::toPython( ctrl->lower_alarm_limit );
      case 'A' :
        return pypdsdata::TypeLib::toPython( ctrl->upper_alarm_limit );
      case 'w' :
        return pypdsdata::TypeLib::toPython( ctrl->lower_warning_limit );
      case 'W' :
        return pypdsdata::TypeLib::toPython( ctrl->upper_warning_limit );
      case 'c' :
        return pypdsdata::TypeLib::toPython( ctrl->lower_ctrl_limit );
      case 'C' :
        return pypdsdata::TypeLib::toPython( ctrl->upper_ctrl_limit );
      default:
        return 0;
      }

    }

  };

  // standard Python stuff
  PyObject* _repr( PyObject *self );

  // methods
  MEMBER_WRAPPER(pypdsdata::EpicsPvCtrl, iPvId)
  MEMBER_WRAPPER(pypdsdata::EpicsPvCtrl, iDbrType)
  MEMBER_WRAPPER(pypdsdata::EpicsPvCtrl, iNumElements)
  MEMBER_WRAPPER(pypdsdata::EpicsPvCtrl, sPvName)
  PyObject* EpicsPvCtrl_status( PyObject* self, void* );
  PyObject* EpicsPvCtrl_severity( PyObject* self, void* );
  PyObject* EpicsPvCtrl_precision( PyObject* self, void* );
  PyObject* EpicsPvCtrl_units_limits( PyObject* self, void* );
  PyObject* EpicsPvCtrl_no_str( PyObject* self, void* );
  PyObject* EpicsPvCtrl_strs( PyObject* self, void* );
  PyObject* EpicsPvCtrl_value( PyObject* self, void* );
  PyObject* EpicsPvCtrl_values( PyObject* self, void* );
  PyObject* EpicsPvCtrl_getnewargs( PyObject* self, PyObject* );

  PyGetSetDef getset[] = {
    {"iPvId",        iPvId,                0, "Pv Id", 0},
    {"iDbrType",     iDbrType,             0, "Epics Data Type", 0},
    {"iNumElements", iNumElements,         0, "Size of Pv Array", 0},
    {"sPvName",      sPvName,              0, "PV name", 0},
    {"status",       EpicsPvCtrl_status,   0, "Status value", 0},
    {"severity",     EpicsPvCtrl_severity, 0, "Severity value", 0},
    {"precision",    EpicsPvCtrl_precision, 0, "Precision Digits, for non-floating types is None", 0},
    {"units",        EpicsPvCtrl_units_limits,    0, "String, None for ENUM and STRING PV types", (void*)"u"},
    {"upper_disp_limit",  EpicsPvCtrl_units_limits,  0, "Set to None for ENUM and STRING PV types", (void*)"D"},
    {"lower_disp_limit",  EpicsPvCtrl_units_limits,  0, "Set to None for ENUM and STRING PV types", (void*)"d"},
    {"upper_alarm_limit",  EpicsPvCtrl_units_limits,  0, "Set to None for ENUM and STRING PV types", (void*)"A"},
    {"upper_warning_limit",  EpicsPvCtrl_units_limits,  0, "Set to None for ENUM and STRING PV types", (void*)"W"},
    {"lower_warning_limit",  EpicsPvCtrl_units_limits,  0, "Set to None for ENUM and STRING PV types", (void*)"w"},
    {"lower_alarm_limit",  EpicsPvCtrl_units_limits,  0, "Set to None for ENUM and STRING PV types", (void*)"a"},
    {"upper_ctrl_limit",  EpicsPvCtrl_units_limits,  0, "Set to None for ENUM and STRING PV types", (void*)"C"},
    {"lower_ctrl_limit",  EpicsPvCtrl_units_limits,  0, "Set to None for ENUM and STRING PV types", (void*)"c"},
    {"no_str",       EpicsPvCtrl_no_str,   0, "Number of ENUM states, None for non-enum PV types", 0},
    {"strs",         EpicsPvCtrl_strs,     0, "List of ENUM states, None for non-enum PV types", 0},
    {"value",        EpicsPvCtrl_value,    0, "PV value, always a single value, for arrays it is first element", 0},
    {"values",       EpicsPvCtrl_values,   0, "List of PV values of size [iNumElements]", 0},
    {0, 0, 0, 0, 0}
  };

  PyMethodDef methods[] = {
    { "__getnewargs__",    EpicsPvCtrl_getnewargs, METH_NOARGS, "Pickle support" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EpicsPvCtrl class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::EpicsPvCtrl::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_getset = ::getset;
  type->tp_str = _repr ;
  type->tp_repr = _repr ;

  BaseType::initType( "EpicsPvCtrl", module );
}


namespace {

template <int iDbrType>
inline
void
print(std::ostream& out, const Pds::EpicsPvCtrlHeader& header)
{
  typedef Pds::EpicsPvCtrl<iDbrType> PVType;
  const PVType& obj = static_cast<const PVType&>(header);
  out << "id=" << obj.iPvId
      << ", name=" << obj.sPvName
      << ", type=" << Pds::Epics::dbr_text[obj.iDbrType]
      << ", status=" << obj.status
      << ", severity=" << obj.severity
      << ", value=" << obj.value;
}


PyObject*
_repr( PyObject *self )
{
  Pds::EpicsPvCtrlHeader* obj = pypdsdata::EpicsPvCtrl::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "EpicsPvCtrl(";
  
  switch ( obj->iDbrType ) {

  case DBR_CTRL_STRING:
    print<DBR_STRING>(str, *obj);
    break;

  case DBR_CTRL_SHORT:
    print<DBR_SHORT>(str, *obj);
    break;

  case DBR_CTRL_FLOAT:
    print<DBR_FLOAT>(str, *obj);
    break;

  case DBR_CTRL_ENUM:
    print<DBR_ENUM>(str, *obj);
    break;

  case DBR_CTRL_CHAR:
    print<DBR_CHAR>(str, *obj);
    break;

  case DBR_CTRL_LONG:
    print<DBR_LONG>(str, *obj);
    break;

  case DBR_CTRL_DOUBLE:
    print<DBR_DOUBLE>(str, *obj);
    break;

  default:
    str << "id=" << obj->iPvId;
  }

  str << ")";
  
  return PyString_FromString( str.str().c_str() );
}

PyObject*
EpicsPvCtrl_status( PyObject* self, void* )
{
  pypdsdata::EpicsPvCtrl* py_this = (pypdsdata::EpicsPvCtrl*) self;
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  // all DBR_CTRL_* types share the same layout for this field, cast to arbitrary type
  typedef Pds::EpicsPvCtrl<DBR_SHORT> CtrlType ;
  CtrlType* pvCtrl = (CtrlType*)py_this->m_obj;

  return pypdsdata::TypeLib::toPython( pvCtrl->status );
}

PyObject*
EpicsPvCtrl_severity( PyObject* self, void* )
{
  pypdsdata::EpicsPvCtrl* py_this = static_cast<pypdsdata::EpicsPvCtrl*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  // all DBR_CTRL_* types share the same layout for this field, cast to arbitrary type
  typedef Pds::EpicsPvCtrl<DBR_SHORT> CtrlType ;
  CtrlType* pvCtrl = (CtrlType*)py_this->m_obj;

  return pypdsdata::TypeLib::toPython( pvCtrl->severity );
}

PyObject*
EpicsPvCtrl_precision( PyObject* self, void* )
{
  pypdsdata::EpicsPvCtrl* py_this = static_cast<pypdsdata::EpicsPvCtrl*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  int precision = 0;
  switch ( py_this->m_obj->iDbrType ) {
  case DBR_CTRL_DOUBLE:
    precision = ((Pds::EpicsPvCtrl<DBR_DOUBLE>*)py_this->m_obj)->precision;
    break;

  case DBR_CTRL_FLOAT:
    precision = ((Pds::EpicsPvCtrl<DBR_FLOAT>*)py_this->m_obj)->precision;
    break;

  default:
    Py_RETURN_NONE;
    break;
  }

  return pypdsdata::TypeLib::toPython( precision );
}

PyObject*
EpicsPvCtrl_units_limits( PyObject* self, void* closure )
{
  pypdsdata::EpicsPvCtrl* py_this = static_cast<pypdsdata::EpicsPvCtrl*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  PyObject* obj = 0;
  switch ( py_this->m_obj->iDbrType ) {
  case DBR_CTRL_SHORT:
    obj = LimitUnitGetter<DBR_SHORT>::get( py_this->m_obj, closure );
    break;

  case DBR_CTRL_CHAR:
    obj = LimitUnitGetter<DBR_CHAR>::get( py_this->m_obj, closure );
    break;

  case DBR_CTRL_LONG:
    obj = LimitUnitGetter<DBR_LONG>::get( py_this->m_obj, closure );
    break;

  case DBR_CTRL_DOUBLE:
    obj = LimitUnitGetter<DBR_DOUBLE>::get( py_this->m_obj, closure );
    break;

  case DBR_CTRL_FLOAT:
    obj = LimitUnitGetter<DBR_FLOAT>::get( py_this->m_obj, closure );
    break;

  }

  if ( obj ) {
    return obj;
  } else {
    Py_RETURN_NONE;
  }
}

/// return the list of enums
PyObject*
EpicsPvCtrl_no_str( PyObject* self, void* )
{
  pypdsdata::EpicsPvCtrl* py_this = static_cast<pypdsdata::EpicsPvCtrl*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  if ( py_this->m_obj->iDbrType == DBR_CTRL_ENUM ) {
    int no_str = ((Pds::EpicsPvCtrl<DBR_ENUM>*)py_this->m_obj)->no_str;
    return pypdsdata::TypeLib::toPython( no_str );
  } else {
    Py_RETURN_NONE;
  }
}

PyObject*
EpicsPvCtrl_strs( PyObject* self, void* )
{
  pypdsdata::EpicsPvCtrl* py_this = static_cast<pypdsdata::EpicsPvCtrl*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  if ( py_this->m_obj->iDbrType == DBR_CTRL_ENUM ) {

    Pds::EpicsPvCtrl<DBR_ENUM>* enumCtrl = (Pds::EpicsPvCtrl<DBR_ENUM>*)py_this->m_obj ;
    int size = enumCtrl->no_str;
    PyObject* list = PyList_New( size );

    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, PyString_FromString( enumCtrl->strs[i] ) );
    }

    return list;

  } else {

    Py_RETURN_NONE;

  }
}

/// return one item from the value array as Python object
template <int iDbrType>
PyObject*
getValue( Pds::EpicsPvCtrlHeader* header, int index = 0 )
{
  typedef Pds::EpicsPvCtrl<iDbrType> CtrlType ;
  CtrlType* obj = static_cast<CtrlType*>(header);
  return pypdsdata::TypeLib::toPython( (&obj->value)[index] );
}

/// Get the first item from the value array
PyObject*
EpicsPvCtrl_value( PyObject* self, void* )
{
  pypdsdata::EpicsPvCtrl* py_this = (pypdsdata::EpicsPvCtrl*) self;
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

  case DBR_CTRL_STRING:
    obj = getValue<DBR_STRING>( py_this->m_obj );
    break;

  case DBR_CTRL_SHORT:
    obj = getValue<DBR_SHORT>( py_this->m_obj );
    break;

  case DBR_CTRL_FLOAT:
    obj = getValue<DBR_FLOAT>( py_this->m_obj );
    break;

  case DBR_CTRL_ENUM:
    obj = getValue<DBR_ENUM>( py_this->m_obj );
    break;

  case DBR_CTRL_CHAR:
    obj = getValue<DBR_CHAR>( py_this->m_obj );
    break;

  case DBR_CTRL_LONG:
    obj = getValue<DBR_LONG>( py_this->m_obj );
    break;

  case DBR_CTRL_DOUBLE:
    obj = getValue<DBR_DOUBLE>( py_this->m_obj );
    break;

  default:
    PyErr_SetString(PyExc_TypeError, "Unexpected PV type");
  }
  return obj;
}

/// get the whole value array
PyObject*
EpicsPvCtrl_values( PyObject* self, void* )
{
  pypdsdata::EpicsPvCtrl* py_this = static_cast<pypdsdata::EpicsPvCtrl*>(self);
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

  case DBR_CTRL_STRING:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<DBR_STRING>( py_this->m_obj ) );
    }
    break;

  case DBR_CTRL_SHORT:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<DBR_SHORT>( py_this->m_obj ) );
    }
    break;

  case DBR_CTRL_FLOAT:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<DBR_FLOAT>( py_this->m_obj ) );
    }
    break;

  case DBR_CTRL_ENUM:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<DBR_ENUM>( py_this->m_obj ) );
    }
    break;

  case DBR_CTRL_CHAR:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<DBR_CHAR>( py_this->m_obj ) );
    }
    break;

  case DBR_CTRL_LONG:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<DBR_LONG>( py_this->m_obj ) );
    }
    break;

  case DBR_CTRL_DOUBLE:
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
EpicsPvCtrl_getnewargs( PyObject* self, PyObject* )
{
  pypdsdata::EpicsPvCtrl* py_this = static_cast<pypdsdata::EpicsPvCtrl*>(self);
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
