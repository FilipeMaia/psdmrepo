//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpicsModule...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "EpicsModule.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ConfigV1.h"
#include "EpicsPvCtrl.h"
#include "EpicsPvTime.h"
#include "epicsTimeStamp.h"
#include "PvConfigV1.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  PyObject* Epics_dbr_type_is_TIME( PyObject*, PyObject* args );
  PyObject* Epics_dbr_type_is_CTRL( PyObject*, PyObject* args );
  PyObject* Epics_from_buffer( PyObject*, PyObject* args );

  PyMethodDef methods[] = {
    {"dbr_type_is_TIME", Epics_dbr_type_is_TIME,  METH_VARARGS,  "dbr_type_is_TIME(typeid: int) -> bool\n\nReturns true for DBR_TIME type IDs." },
    {"dbr_type_is_CTRL", Epics_dbr_type_is_CTRL,  METH_VARARGS,  "dbr_type_is_CTRL(typeid: int) -> bool\n\nReturns true for DBR_CTRL type IDs." },
    {"from_buffer", Epics_from_buffer,  METH_VARARGS,  "from_buffer(buffer) -> object\n\nBuild EPICS object from memory buffer." },
    {0, 0, 0, 0}
   };

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace pypdsdata {
namespace Epics {

PyObject* EpicsModule::s_module = 0;

PyObject*
EpicsModule::getModule()
{
  if ( s_module ) return s_module;

  // create the module
  PyObject* module = Py_InitModule3( "_pdsdata.epics", methods, "The Python wrapper module for pdsdata/epics" );

  // define constants
  PyModule_AddIntConstant( module, "DBR_STRING", Pds::Epics::DBR_STRING );
  PyModule_AddIntConstant( module, "DBR_SHORT", Pds::Epics::DBR_SHORT );
  PyModule_AddIntConstant( module, "DBR_FLOAT", Pds::Epics::DBR_FLOAT );
  PyModule_AddIntConstant( module, "DBR_ENUM", Pds::Epics::DBR_ENUM );
  PyModule_AddIntConstant( module, "DBR_CHAR", Pds::Epics::DBR_CHAR );
  PyModule_AddIntConstant( module, "DBR_LONG", Pds::Epics::DBR_LONG );
  PyModule_AddIntConstant( module, "DBR_DOUBLE", Pds::Epics::DBR_DOUBLE );
  PyModule_AddIntConstant( module, "DBR_STS_STRING", Pds::Epics::DBR_STS_STRING );
  PyModule_AddIntConstant( module, "DBR_STS_SHORT", Pds::Epics::DBR_STS_SHORT );
  PyModule_AddIntConstant( module, "DBR_STS_FLOAT", Pds::Epics::DBR_STS_FLOAT );
  PyModule_AddIntConstant( module, "DBR_STS_ENUM", Pds::Epics::DBR_STS_ENUM );
  PyModule_AddIntConstant( module, "DBR_STS_CHAR", Pds::Epics::DBR_STS_CHAR );
  PyModule_AddIntConstant( module, "DBR_STS_LONG", Pds::Epics::DBR_STS_LONG );
  PyModule_AddIntConstant( module, "DBR_STS_DOUBLE", Pds::Epics::DBR_STS_DOUBLE );
  PyModule_AddIntConstant( module, "DBR_TIME_STRING", Pds::Epics::DBR_TIME_STRING );
  PyModule_AddIntConstant( module, "DBR_TIME_INT", Pds::Epics::DBR_TIME_INT );
  PyModule_AddIntConstant( module, "DBR_TIME_SHORT", Pds::Epics::DBR_TIME_SHORT );
  PyModule_AddIntConstant( module, "DBR_TIME_FLOAT", Pds::Epics::DBR_TIME_FLOAT );
  PyModule_AddIntConstant( module, "DBR_TIME_ENUM", Pds::Epics::DBR_TIME_ENUM );
  PyModule_AddIntConstant( module, "DBR_TIME_CHAR", Pds::Epics::DBR_TIME_CHAR );
  PyModule_AddIntConstant( module, "DBR_TIME_LONG", Pds::Epics::DBR_TIME_LONG );
  PyModule_AddIntConstant( module, "DBR_TIME_DOUBLE", Pds::Epics::DBR_TIME_DOUBLE );
  PyModule_AddIntConstant( module, "DBR_GR_STRING", Pds::Epics::DBR_GR_STRING );
  PyModule_AddIntConstant( module, "DBR_GR_SHORT", Pds::Epics::DBR_GR_SHORT );
  PyModule_AddIntConstant( module, "DBR_GR_FLOAT", Pds::Epics::DBR_GR_FLOAT );
  PyModule_AddIntConstant( module, "DBR_GR_ENUM", Pds::Epics::DBR_GR_ENUM );
  PyModule_AddIntConstant( module, "DBR_GR_CHAR", Pds::Epics::DBR_GR_CHAR );
  PyModule_AddIntConstant( module, "DBR_GR_LONG", Pds::Epics::DBR_GR_LONG );
  PyModule_AddIntConstant( module, "DBR_GR_DOUBLE", Pds::Epics::DBR_GR_DOUBLE );
  PyModule_AddIntConstant( module, "DBR_CTRL_STRING", Pds::Epics::DBR_CTRL_STRING );
  PyModule_AddIntConstant( module, "DBR_CTRL_SHORT", Pds::Epics::DBR_CTRL_SHORT );
  PyModule_AddIntConstant( module, "DBR_CTRL_FLOAT", Pds::Epics::DBR_CTRL_FLOAT );
  PyModule_AddIntConstant( module, "DBR_CTRL_ENUM", Pds::Epics::DBR_CTRL_ENUM );
  PyModule_AddIntConstant( module, "DBR_CTRL_CHAR", Pds::Epics::DBR_CTRL_CHAR );
  PyModule_AddIntConstant( module, "DBR_CTRL_LONG", Pds::Epics::DBR_CTRL_LONG );
  PyModule_AddIntConstant( module, "DBR_CTRL_DOUBLE", Pds::Epics::DBR_CTRL_DOUBLE );

  // add types
  pypdsdata::Epics::EpicsPvCtrl::initType( module );
  pypdsdata::Epics::EpicsPvTime::initType( module );
  pypdsdata::Epics::ConfigV1::initType( module );
  pypdsdata::Epics::PvConfigV1::initType( module );
  pypdsdata::Epics::epicsTimeStamp::initType( module );

//  // make the list of severity strings
//  PyObject* strList = PyList_New(ALARM_NSEV);
//  for ( int i = 0 ; i < ALARM_NSEV ; ++ i ) {
//    PyList_SET_ITEM( strList, i, PyString_FromString( Pds::Epics::epicsAlarmSeverityStrings[i] ) );
//  }
//  Py_INCREF( strList );
//  PyModule_AddObject( module, "epicsAlarmSeverityStrings", strList );
//
//  // make the list of conditions strings
//  strList = PyList_New(ALARM_NSTATUS);
//  for ( int i = 0 ; i < ALARM_NSTATUS ; ++ i ) {
//    PyList_SET_ITEM( strList, i, PyString_FromString( Pds::Epics::epicsAlarmConditionStrings[i] ) );
//  }
//  Py_INCREF( strList );
//  PyModule_AddObject( module, "epicsAlarmConditionStrings", strList );
//
//  // make the list of conditions strings
//  const int ndbr = sizeof Pds::Epics::dbr_text / sizeof Pds::Epics::dbr_text[0] ;
//  strList = PyList_New( ndbr );
//  for ( int i = 0 ; i <  ndbr ; ++ i ) {
//    PyList_SET_ITEM( strList, i, PyString_FromString( Pds::Epics::dbr_text[i] ) );
//  }
//  Py_INCREF( strList );
//  PyModule_AddObject( module, "dbr_text", strList );

  // store it
  s_module = module ;

  return s_module;
}

// make Python object from Pds type
PyObject*
EpicsModule::PyObject_FromPds( Pds::Epics::EpicsPvHeader* pvHeader, PyObject* parent, size_t size )
{
  if ( pvHeader->isTime() ) {
    return pypdsdata::Epics::EpicsPvTime::PyObject_FromPds( static_cast<Pds::Epics::EpicsPvTimeHeader*>(pvHeader), parent, size );
  } else if ( pvHeader->isCtrl() ) {
    return pypdsdata::Epics::EpicsPvCtrl::PyObject_FromPds( static_cast<Pds::Epics::EpicsPvCtrlHeader*>(pvHeader), parent, size);
  } else {
    PyErr_SetString(PyExc_TypeError, "Unknown EPICS PV type");
    return 0;
  }
}

} // namespace Epics
} // namespace pypdsdata


namespace {

PyObject*
Epics_dbr_type_is_TIME( PyObject*, PyObject* args )
{
  int id = 0;
  if ( not PyArg_ParseTuple( args, "I:epics.dbr_type_is_TIME", &id ) ) return 0;

  return PyBool_FromLong( id >= Pds::Epics::DBR_TIME_STRING and id <= Pds::Epics::DBR_TIME_DOUBLE );
}

PyObject*
Epics_dbr_type_is_CTRL( PyObject*, PyObject* args )
{
  int id = 0;
  if ( not PyArg_ParseTuple( args, "I:epics.dbr_type_is_CTRL", &id ) ) return 0;

  return PyBool_FromLong( id >= Pds::Epics::DBR_CTRL_STRING and id <= Pds::Epics::DBR_CTRL_DOUBLE );
}

PyObject*
Epics_from_buffer( PyObject*, PyObject* args )
{
  // parse arguments must be a buffer object
  PyObject* parent = PyTuple_GetItem(args, 0);
  const char* buf;
  int bufsize;
  if ( not PyArg_ParseTuple( args, "s#:pypdsdata::Dgram", &buf, &bufsize ) ) return 0;

  // buffer must contain valid memory representation of Epics data
  Pds::Epics::EpicsPvHeader* pvHeader = (Pds::Epics::EpicsPvHeader*)buf;

  if ( pvHeader->isTime() ) {
    return pypdsdata::Epics::EpicsPvTime::PyObject_FromPds( static_cast<Pds::Epics::EpicsPvTimeHeader*>(pvHeader), parent, bufsize );
  } else if ( pvHeader->isCtrl() ) {
    return pypdsdata::Epics::EpicsPvCtrl::PyObject_FromPds( static_cast<Pds::Epics::EpicsPvCtrlHeader*>(pvHeader), parent, bufsize );
  } else {
    PyErr_SetString(PyExc_TypeError, "Unknown EPICS PV type");
    return 0;
  }
}

}
