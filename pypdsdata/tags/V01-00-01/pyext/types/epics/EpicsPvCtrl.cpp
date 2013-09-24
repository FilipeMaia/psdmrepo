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

  template <typename EpicsType>
  struct LimitUnitGetter {

    static PyObject* get( const Pds::Epics::EpicsPvCtrlHeader& header, void* closure ) {

      using pypdsdata::TypeLib::toPython;
      
      const EpicsType& ctrl = static_cast<const EpicsType&>(header);
      switch( *(char*)closure ) {
      case 'u' :
        return toPython( ctrl.dbr().units() );
      case 'd' :
        return toPython( ctrl.dbr().lower_disp_limit() );
      case 'D' :
        return toPython( ctrl.dbr().upper_disp_limit() );
      case 'a' :
        return toPython( ctrl.dbr().lower_alarm_limit() );
      case 'A' :
        return toPython( ctrl.dbr().upper_alarm_limit() );
      case 'w' :
        return toPython( ctrl.dbr().lower_warning_limit() );
      case 'W' :
        return toPython( ctrl.dbr().upper_warning_limit() );
      case 'c' :
        return toPython( ctrl.dbr().lower_ctrl_limit() );
      case 'C' :
        return toPython( ctrl.dbr().upper_ctrl_limit() );
      default:
        return 0;
      }

    }

  };

  template <typename EpicsType>
  inline
  void
  print(std::ostream& out, const Pds::Epics::EpicsPvCtrlHeader& header)
  {
    const EpicsType& obj = static_cast<const EpicsType&>(header);
    out << "id=" << obj.pvId()
        << ", name=" << obj.pvName()
        << ", type=" << obj.dbrType()
        << ", status=" << obj.status()
        << ", severity=" << obj.severity()
        << ", value=" << obj.value(0);
  }

  // methods
  namespace gs {
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Epics::EpicsPvCtrl, pvId)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Epics::EpicsPvCtrl, dbrType)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Epics::EpicsPvCtrl, numElements)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Epics::EpicsPvCtrl, pvName)
  PyObject* EpicsPvCtrl_status( PyObject* self, void* );
  PyObject* EpicsPvCtrl_severity( PyObject* self, void* );
  PyObject* EpicsPvCtrl_precision( PyObject* self, void* );
  PyObject* EpicsPvCtrl_units_limits( PyObject* self, void* );
  PyObject* EpicsPvCtrl_no_str( PyObject* self, void* );
  PyObject* EpicsPvCtrl_strs( PyObject* self, void* );
  PyObject* EpicsPvCtrl_value( PyObject* self, void* );
  PyObject* EpicsPvCtrl_values( PyObject* self, void* );
  }


  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"iPvId",                   gs::pvId,                       0, "Integer number, Pv Id", 0},
    {"iDbrType",                gs::dbrType,                    0, "Integer number, Epics Data Type", 0},
    {"iNumElements",            gs::numElements,                0, "Integer number, Size of Pv Array", 0},
    {"sPvName",                 gs::pvName,                     0, "String, PV name", 0},
    {"status",                  gs::EpicsPvCtrl_status,         0, "Integer number, Status value", 0},
    {"severity",                gs::EpicsPvCtrl_severity,       0, "Integer number, Severity value", 0},
    {"precision",               gs::EpicsPvCtrl_precision,      0, "Integer number, Precision Digits, for non-floating types is None", 0},
    {"units",                   gs::EpicsPvCtrl_units_limits,   0, "String, None for ENUM and STRING PV types", (void*)"u"},
    {"upper_disp_limit",        gs::EpicsPvCtrl_units_limits,   0, "Number, None for ENUM and STRING PV types", (void*)"D"},
    {"lower_disp_limit",        gs::EpicsPvCtrl_units_limits,   0, "Number, None for ENUM and STRING PV types", (void*)"d"},
    {"upper_alarm_limit",       gs::EpicsPvCtrl_units_limits,   0, "Number, None for ENUM and STRING PV types", (void*)"A"},
    {"upper_warning_limit",     gs::EpicsPvCtrl_units_limits,   0, "Number, None for ENUM and STRING PV types", (void*)"W"},
    {"lower_warning_limit",     gs::EpicsPvCtrl_units_limits,   0, "Number, None for ENUM and STRING PV types", (void*)"w"},
    {"lower_alarm_limit",       gs::EpicsPvCtrl_units_limits,   0, "Number, None for ENUM and STRING PV types", (void*)"a"},
    {"upper_ctrl_limit",        gs::EpicsPvCtrl_units_limits,   0, "Number, None for ENUM and STRING PV types", (void*)"C"},
    {"lower_ctrl_limit",        gs::EpicsPvCtrl_units_limits,   0, "Number, None for ENUM and STRING PV types", (void*)"c"},
    {"no_str",                  gs::EpicsPvCtrl_no_str,         0, "Number of ENUM states, None for non-enum PV types", 0},
    {"strs",                    gs::EpicsPvCtrl_strs,           0, "List of ENUM states, None for non-enum PV types", 0},
    {"value",                   gs::EpicsPvCtrl_value,          0, "PV value (number or string), always a single value, for arrays it is first element", 0},
    {"values",                  gs::EpicsPvCtrl_values,         0, "List of PV values of size [iNumElements]", 0},
    {0, 0, 0, 0, 0}
  };

  namespace mm {
  PyObject* EpicsPvCtrl_getnewargs( PyObject* self, PyObject* );
  }
  
  PyMethodDef methods[] = {
    { "__getnewargs__",    mm::EpicsPvCtrl_getnewargs, METH_NOARGS, "Pickle support" },
    {0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::EpicsPvCtrl class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Epics::EpicsPvCtrl::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_getset = ::getset;

  BaseType::initType( "EpicsPvCtrl", module );
}

void
pypdsdata::Epics::EpicsPvCtrl::print(std::ostream& str) const
{
  str << "EpicsPvCtrl(";
  
  switch ( m_obj->dbrType() ) {

  case Pds::Epics::DBR_CTRL_STRING:
    ::print<Pds::Epics::EpicsPvCtrlString>(str, *m_obj);
    break;

  case Pds::Epics::DBR_CTRL_SHORT:
    ::print<Pds::Epics::EpicsPvCtrlShort>(str, *m_obj);
    break;

  case Pds::Epics::DBR_CTRL_FLOAT:
    ::print<Pds::Epics::EpicsPvCtrlFloat>(str, *m_obj);
    break;

  case Pds::Epics::DBR_CTRL_ENUM:
    ::print<Pds::Epics::EpicsPvCtrlEnum>(str, *m_obj);
    break;

  case Pds::Epics::DBR_CTRL_CHAR:
    ::print<Pds::Epics::EpicsPvCtrlChar>(str, *m_obj);
    break;

  case Pds::Epics::DBR_CTRL_LONG:
    ::print<Pds::Epics::EpicsPvCtrlLong>(str, *m_obj);
    break;

  case Pds::Epics::DBR_CTRL_DOUBLE:
    ::print<Pds::Epics::EpicsPvCtrlDouble>(str, *m_obj);
    break;

  default:
    str << "id=" << m_obj->pvId();
  }

  str << ")";
}

namespace {

PyObject*
gs::EpicsPvCtrl_status( PyObject* self, void* )
{
  const Pds::Epics::EpicsPvCtrlHeader* obj = pypdsdata::Epics::EpicsPvCtrl::pdsObject( self );
  if ( not obj ) return 0;

  // all DBR_CTRL_* types share the same layout for this field, cast to arbitrary type
  typedef Pds::Epics::EpicsPvCtrlShort CtrlType ;
  const CtrlType& pvCtrl = static_cast<const CtrlType&>(*obj);

  using pypdsdata::TypeLib::toPython;
  return toPython( pvCtrl.status() );
}

PyObject*
gs::EpicsPvCtrl_severity( PyObject* self, void* )
{
  const Pds::Epics::EpicsPvCtrlHeader* obj = pypdsdata::Epics::EpicsPvCtrl::pdsObject( self );
  if ( not obj ) return 0;

  // all DBR_CTRL_* types share the same layout for this field, cast to arbitrary type
  typedef Pds::Epics::EpicsPvCtrlShort CtrlType ;
  const CtrlType& pvCtrl = static_cast<const CtrlType&>(*obj);

  using pypdsdata::TypeLib::toPython;
  return toPython( pvCtrl.severity() );
}

PyObject*
gs::EpicsPvCtrl_precision( PyObject* self, void* )
{
  const Pds::Epics::EpicsPvCtrlHeader* obj = pypdsdata::Epics::EpicsPvCtrl::pdsObject( self );
  if ( not obj ) return 0;

  int precision = 0;
  switch ( obj->dbrType() ) {
  case Pds::Epics::DBR_CTRL_DOUBLE:
    precision = static_cast<const Pds::Epics::EpicsPvCtrlDouble&>(*obj).dbr().precision();
    break;

  case Pds::Epics::DBR_CTRL_FLOAT:
    precision = static_cast<const Pds::Epics::EpicsPvCtrlFloat&>(*obj).dbr().precision();
    break;

  default:
    Py_RETURN_NONE;
    break;
  }

  using pypdsdata::TypeLib::toPython;
  return toPython( precision );
}

PyObject*
gs::EpicsPvCtrl_units_limits( PyObject* self, void* closure )
{
  const Pds::Epics::EpicsPvCtrlHeader* obj = pypdsdata::Epics::EpicsPvCtrl::pdsObject( self );
  if ( not obj ) return 0;

  PyObject* pyobj = 0;
  switch ( obj->dbrType() ) {
  case Pds::Epics::DBR_CTRL_SHORT:
    pyobj = LimitUnitGetter<Pds::Epics::EpicsPvCtrlShort>::get( *obj, closure );
    break;

  case Pds::Epics::DBR_CTRL_CHAR:
    pyobj = LimitUnitGetter<Pds::Epics::EpicsPvCtrlChar>::get( *obj, closure );
    break;

  case Pds::Epics::DBR_CTRL_LONG:
    pyobj = LimitUnitGetter<Pds::Epics::EpicsPvCtrlLong>::get( *obj, closure );
    break;

  case Pds::Epics::DBR_CTRL_DOUBLE:
    pyobj = LimitUnitGetter<Pds::Epics::EpicsPvCtrlDouble>::get( *obj, closure );
    break;

  case Pds::Epics::DBR_CTRL_FLOAT:
    pyobj = LimitUnitGetter<Pds::Epics::EpicsPvCtrlFloat>::get( *obj, closure );
    break;

  }

  if ( pyobj ) {
    return pyobj;
  } else {
    Py_RETURN_NONE;
  }
}

/// return the list of enums
PyObject*
gs::EpicsPvCtrl_no_str( PyObject* self, void* )
{
  const Pds::Epics::EpicsPvCtrlHeader* obj = pypdsdata::Epics::EpicsPvCtrl::pdsObject( self );
  if ( not obj ) return 0;

  if ( obj->dbrType() == Pds::Epics::DBR_CTRL_ENUM ) {
    int no_str = static_cast<const Pds::Epics::EpicsPvCtrlEnum*>(obj)->dbr().no_str();
    return pypdsdata::TypeLib::toPython( no_str );
  } else {
    Py_RETURN_NONE;
  }
}

PyObject*
gs::EpicsPvCtrl_strs( PyObject* self, void* )
{
  const Pds::Epics::EpicsPvCtrlHeader* obj = pypdsdata::Epics::EpicsPvCtrl::pdsObject( self );
  if ( not obj ) return 0;

  if ( obj->dbrType() == Pds::Epics::DBR_CTRL_ENUM ) {

    const Pds::Epics::EpicsPvCtrlEnum* enumCtrl = static_cast<const Pds::Epics::EpicsPvCtrlEnum*>(obj);
    int size = enumCtrl->dbr().no_str();
    PyObject* list = PyList_New( size );

    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, PyString_FromString( enumCtrl->dbr().strings(i) ) );
    }

    return list;

  } else {

    Py_RETURN_NONE;

  }
}

/// return one item from the value array as Python object
template <typename EpicsType>
PyObject*
getValue(const Pds::Epics::EpicsPvCtrlHeader& header, int index = 0)
{
  const EpicsType& ctrl = static_cast<const EpicsType&>(header);
  using pypdsdata::TypeLib::toPython;
  return toPython(ctrl.value(index));
}

/// Get the first item from the value array
PyObject*
gs::EpicsPvCtrl_value( PyObject* self, void* )
{
  const Pds::Epics::EpicsPvCtrlHeader* obj = pypdsdata::Epics::EpicsPvCtrl::pdsObject( self );
  if ( not obj ) return 0;

  if ( obj->numElements() <= 0 ) {
    PyErr_SetString(PyExc_TypeError, "Non-positive PV array size");
    return 0;
  }

  PyObject* pyobj = 0;
  switch ( obj->dbrType() ) {

  case Pds::Epics::DBR_CTRL_STRING:
    pyobj = getValue<Pds::Epics::EpicsPvCtrlString>( *obj );
    break;

  case Pds::Epics::DBR_CTRL_SHORT:
    pyobj = getValue<Pds::Epics::EpicsPvCtrlShort>( *obj );
    break;

  case Pds::Epics::DBR_CTRL_FLOAT:
    pyobj = getValue<Pds::Epics::EpicsPvCtrlFloat>( *obj );
    break;

  case Pds::Epics::DBR_CTRL_ENUM:
    pyobj = getValue<Pds::Epics::EpicsPvCtrlEnum>( *obj );
    break;

  case Pds::Epics::DBR_CTRL_CHAR:
    pyobj = getValue<Pds::Epics::EpicsPvCtrlChar>( *obj );
    break;

  case Pds::Epics::DBR_CTRL_LONG:
    pyobj = getValue<Pds::Epics::EpicsPvCtrlLong>( *obj );
    break;

  case Pds::Epics::DBR_CTRL_DOUBLE:
    pyobj = getValue<Pds::Epics::EpicsPvCtrlDouble>( *obj );
    break;

  default:
    PyErr_SetString(PyExc_TypeError, "Unexpected PV type");
  }
  return pyobj;
}

/// get the whole value array
PyObject*
gs::EpicsPvCtrl_values( PyObject* self, void* )
{
  const Pds::Epics::EpicsPvCtrlHeader* obj = pypdsdata::Epics::EpicsPvCtrl::pdsObject( self );
  if ( not obj ) return 0;

  int size = obj->numElements();
  if ( size < 0 ) {
    PyErr_SetString(PyExc_TypeError, "Negative PV array size");
    return 0;
  }

  PyObject* list = PyList_New( size );
  switch ( obj->dbrType() ) {

  case Pds::Epics::DBR_CTRL_STRING:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<Pds::Epics::EpicsPvCtrlString>( *obj ) );
    }
    break;

  case Pds::Epics::DBR_CTRL_SHORT:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<Pds::Epics::EpicsPvCtrlShort>( *obj ) );
    }
    break;

  case Pds::Epics::DBR_CTRL_FLOAT:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<Pds::Epics::EpicsPvCtrlFloat>( *obj ) );
    }
    break;

  case Pds::Epics::DBR_CTRL_ENUM:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<Pds::Epics::EpicsPvCtrlEnum>( *obj ) );
    }
    break;

  case Pds::Epics::DBR_CTRL_CHAR:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<Pds::Epics::EpicsPvCtrlChar>( *obj ) );
    }
    break;

  case Pds::Epics::DBR_CTRL_LONG:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<Pds::Epics::EpicsPvCtrlLong>( *obj ) );
    }
    break;

  case Pds::Epics::DBR_CTRL_DOUBLE:
    for ( int i = 0 ; i < size ; ++ i ) {
      PyList_SET_ITEM( list, i, getValue<Pds::Epics::EpicsPvCtrlDouble>( *obj ) );
    }
    break;

  default:
    PyErr_SetString(PyExc_TypeError, "Unexpected PV type");
  }
  return list;
}

PyObject*
mm::EpicsPvCtrl_getnewargs( PyObject* self, PyObject* )
{
  pypdsdata::Epics::EpicsPvCtrl* py_this = static_cast<pypdsdata::Epics::EpicsPvCtrl*>(self);
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
