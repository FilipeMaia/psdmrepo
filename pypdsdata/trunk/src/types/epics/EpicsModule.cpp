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
#include "EpicsPvCtrl.h"
#include "EpicsPvTime.h"
#include "epicsTimeStamp.h"
#include "pdsdata/epics/EpicsDbrTools.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  PyObject* Epics_dbr_type_is_TIME( PyObject*, PyObject* args );
  PyObject* Epics_dbr_type_is_CTRL( PyObject*, PyObject* args );

  PyMethodDef methods[] = {
    {"dbr_type_is_TIME", Epics_dbr_type_is_TIME,  METH_VARARGS,  "Returns true for DBR_TIME type IDs." },
    {"dbr_type_is_CTRL", Epics_dbr_type_is_CTRL,  METH_VARARGS,  "Returns true for DBR_CTRL type IDs." },
    {0, 0, 0, 0}
   };

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace pypdsdata {

PyObject* EpicsModule::s_module = 0;

PyObject*
EpicsModule::getModule()
{
  if ( s_module ) return s_module;

  // create the module
  PyObject* module = Py_InitModule3( "_pdsdata.epics", methods, "The Python module for pdsdata/epics" );

  // define constants
  PyModule_AddIntConstant( module, "DBR_STRING", DBR_STRING );
  PyModule_AddIntConstant( module, "DBR_SHORT", DBR_SHORT );
  PyModule_AddIntConstant( module, "DBR_FLOAT", DBR_FLOAT );
  PyModule_AddIntConstant( module, "DBR_ENUM", DBR_ENUM );
  PyModule_AddIntConstant( module, "DBR_CHAR", DBR_CHAR );
  PyModule_AddIntConstant( module, "DBR_LONG", DBR_LONG );
  PyModule_AddIntConstant( module, "DBR_DOUBLE", DBR_DOUBLE );
  PyModule_AddIntConstant( module, "DBR_STS_STRING", DBR_STS_STRING );
  PyModule_AddIntConstant( module, "DBR_STS_SHORT", DBR_STS_SHORT );
  PyModule_AddIntConstant( module, "DBR_STS_FLOAT", DBR_STS_FLOAT );
  PyModule_AddIntConstant( module, "DBR_STS_ENUM", DBR_STS_ENUM );
  PyModule_AddIntConstant( module, "DBR_STS_CHAR", DBR_STS_CHAR );
  PyModule_AddIntConstant( module, "DBR_STS_LONG", DBR_STS_LONG );
  PyModule_AddIntConstant( module, "DBR_STS_DOUBLE", DBR_STS_DOUBLE );
  PyModule_AddIntConstant( module, "DBR_TIME_STRING", DBR_TIME_STRING );
  PyModule_AddIntConstant( module, "DBR_TIME_INT", DBR_TIME_INT );
  PyModule_AddIntConstant( module, "DBR_TIME_SHORT", DBR_TIME_SHORT );
  PyModule_AddIntConstant( module, "DBR_TIME_FLOAT", DBR_TIME_FLOAT );
  PyModule_AddIntConstant( module, "DBR_TIME_ENUM", DBR_TIME_ENUM );
  PyModule_AddIntConstant( module, "DBR_TIME_CHAR", DBR_TIME_CHAR );
  PyModule_AddIntConstant( module, "DBR_TIME_LONG", DBR_TIME_LONG );
  PyModule_AddIntConstant( module, "DBR_TIME_DOUBLE", DBR_TIME_DOUBLE );
  PyModule_AddIntConstant( module, "DBR_GR_STRING", DBR_GR_STRING );
  PyModule_AddIntConstant( module, "DBR_GR_SHORT", DBR_GR_SHORT );
  PyModule_AddIntConstant( module, "DBR_GR_FLOAT", DBR_GR_FLOAT );
  PyModule_AddIntConstant( module, "DBR_GR_ENUM", DBR_GR_ENUM );
  PyModule_AddIntConstant( module, "DBR_GR_CHAR", DBR_GR_CHAR );
  PyModule_AddIntConstant( module, "DBR_GR_LONG", DBR_GR_LONG );
  PyModule_AddIntConstant( module, "DBR_GR_DOUBLE", DBR_GR_DOUBLE );
  PyModule_AddIntConstant( module, "DBR_CTRL_STRING", DBR_CTRL_STRING );
  PyModule_AddIntConstant( module, "DBR_CTRL_SHORT", DBR_CTRL_SHORT );
  PyModule_AddIntConstant( module, "DBR_CTRL_FLOAT", DBR_CTRL_FLOAT );
  PyModule_AddIntConstant( module, "DBR_CTRL_ENUM", DBR_CTRL_ENUM );
  PyModule_AddIntConstant( module, "DBR_CTRL_CHAR", DBR_CTRL_CHAR );
  PyModule_AddIntConstant( module, "DBR_CTRL_LONG", DBR_CTRL_LONG );
  PyModule_AddIntConstant( module, "DBR_CTRL_DOUBLE", DBR_CTRL_DOUBLE );

  // add types
  pypdsdata::EpicsPvCtrl::initType( module );
  pypdsdata::EpicsPvTime::initType( module );
  pypdsdata::Epics::epicsTimeStamp::initType( module );

  // make the list of severity strings
  PyObject* strList = PyList_New(ALARM_NSEV);
  for ( int i = 0 ; i < ALARM_NSEV ; ++ i ) {
    PyList_SET_ITEM( strList, i, PyString_FromString( Pds::Epics::epicsAlarmSeverityStrings[i] ) );
  }
  Py_INCREF( strList );
  PyModule_AddObject( module, "epicsAlarmSeverityStrings", strList );

  // make the list of conditions strings
  strList = PyList_New(ALARM_NSTATUS);
  for ( int i = 0 ; i < ALARM_NSTATUS ; ++ i ) {
    PyList_SET_ITEM( strList, i, PyString_FromString( Pds::Epics::epicsAlarmConditionStrings[i] ) );
  }
  Py_INCREF( strList );
  PyModule_AddObject( module, "epicsAlarmConditionStrings", strList );

  // make the list of conditions strings
  const int ndbr = sizeof Pds::Epics::dbr_text / sizeof Pds::Epics::dbr_text[0] ;
  strList = PyList_New( ndbr );
  for ( int i = 0 ; i <  ndbr ; ++ i ) {
    PyList_SET_ITEM( strList, i, PyString_FromString( Pds::Epics::dbr_text[i] ) );
  }
  Py_INCREF( strList );
  PyModule_AddObject( module, "dbr_text", strList );

  // store it
  s_module = module ;

  return s_module;
}

// make Python object from Pds type
PyObject*
EpicsModule::PyObject_FromPds( Pds::EpicsPvHeader* pvHeader, PyObject* parent, size_t size )
{
  if ( dbr_type_is_TIME(pvHeader->iDbrType) ) {
    return EpicsPvTime::PyObject_FromPds( pvHeader, parent, size );
  } else if ( dbr_type_is_CTRL(pvHeader->iDbrType) ) {
    return EpicsPvCtrl::PyObject_FromPds( static_cast<Pds::EpicsPvCtrlHeader*>(pvHeader), parent, size);
  } else {
    PyErr_SetString(PyExc_TypeError, "Unknown EPICS PV type");
    return 0;
  }
}

} // namespace pypdsdata


namespace {

PyObject*
Epics_dbr_type_is_TIME( PyObject*, PyObject* args )
{
  int id = 0;
  if ( not PyArg_ParseTuple( args, "I:epics.dbr_type_is_TIME", &id ) ) return 0;

  return PyBool_FromLong( dbr_type_is_TIME(id) );
}

PyObject*
Epics_dbr_type_is_CTRL( PyObject*, PyObject* args )
{
  int id = 0;
  if ( not PyArg_ParseTuple( args, "I:epics.dbr_type_is_CTRL", &id ) ) return 0;

  return PyBool_FromLong( dbr_type_is_CTRL(id) );
}

}
