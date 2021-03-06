//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV1QuadReg...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV1QuadReg.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CsPadDigitalPotsCfg.h"
#include "CsPadGainMapCfg.h"
#include "CsPadReadOnlyCfg.h"
#include "../../EnumType.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  pypdsdata::EnumType::Enum dataModesEnumValues[] = {
      { "normal",     Pds::CsPad::normal },
      { "shiftTest",  Pds::CsPad::shiftTest },
      { "testData",   Pds::CsPad::testData },
      { "reserved",   Pds::CsPad::reserved },
      { 0, 0 }
  };
  pypdsdata::EnumType dataModesEnum ( "DataModes", dataModesEnumValues );

  // methods
  PyObject* shiftSelect( PyObject* self, PyObject* );
  PyObject* edgeSelect( PyObject* self, PyObject* );
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV1QuadReg, readClkSet)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV1QuadReg, readClkHold)
  ENUM_FUN0_WRAPPER(pypdsdata::CsPad::ConfigV1QuadReg, dataMode, dataModesEnum)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV1QuadReg, prstSel)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV1QuadReg, acqDelay)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV1QuadReg, intTime)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV1QuadReg, digDelay)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV1QuadReg, ampIdle)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV1QuadReg, injTotal)
  FUN0_WRAPPER(pypdsdata::CsPad::ConfigV1QuadReg, rowColShiftPer)
  PyObject* readOnly( PyObject* self, PyObject* );
  PyObject* digitalPots( PyObject* self, PyObject* );
  PyObject* gainMap( PyObject* self, PyObject* );

  PyMethodDef methods[] = {
    {"shiftSelect",     shiftSelect,    METH_NOARGS, "" },
    {"edgeSelect",      edgeSelect,     METH_NOARGS, "" },
    {"readClkSet",      readClkSet,     METH_NOARGS, "" },
    {"readClkHold",     readClkHold,    METH_NOARGS, "" },
    {"dataMode",        dataMode,       METH_NOARGS, "" },
    {"prstSel",         prstSel,        METH_NOARGS, "" },
    {"acqDelay",        acqDelay,       METH_NOARGS, "" },
    {"intTime",         intTime,        METH_NOARGS, "" },
    {"digDelay",        digDelay,       METH_NOARGS, "" },
    {"ampIdle",         ampIdle,        METH_NOARGS, "" },
    {"injTotal",        injTotal,       METH_NOARGS, "" },
    {"rowColShiftPer",  rowColShiftPer, METH_NOARGS, "" },
    {"ro",              readOnly,       METH_NOARGS, "" },
    {"dp",              digitalPots,    METH_NOARGS, "" },
    {"gm",              gainMap,        METH_NOARGS, "" },
    {"readOnly",        readOnly,       METH_NOARGS, "" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::CsPad::ConfigV1QuadReg class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::CsPad::ConfigV1QuadReg::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "DataModes", dataModesEnum.type() );
  PyObject* val = PyInt_FromLong(Pds::CsPad::TwoByTwosPerQuad);
  PyDict_SetItemString( type->tp_dict, "TwoByTwosPerQuad", val );
  Py_XDECREF(val);

  BaseType::initType( "ConfigV1QuadReg", module );
}

namespace {

PyObject*
shiftSelect( PyObject* self, PyObject* )
{
  const Pds::CsPad::ConfigV1QuadReg* obj = pypdsdata::CsPad::ConfigV1QuadReg::pdsObject( self );
  if ( not obj ) return 0;

  PyObject* list = PyList_New( Pds::CsPad::TwoByTwosPerQuad );
  const uint32_t* data = obj->shiftSelect();
  for ( unsigned i = 0 ; i < Pds::CsPad::TwoByTwosPerQuad ; ++ i ) {
    PyList_SET_ITEM( list, i, PyInt_FromLong(data[i]) );
  }

  return list;
}

PyObject*
edgeSelect( PyObject* self, PyObject* )
{
  const Pds::CsPad::ConfigV1QuadReg* obj = pypdsdata::CsPad::ConfigV1QuadReg::pdsObject( self );
  if ( not obj ) return 0;

  PyObject* list = PyList_New( Pds::CsPad::TwoByTwosPerQuad );
  const uint32_t* data = obj->edgeSelect();
  for ( unsigned i = 0 ; i < Pds::CsPad::TwoByTwosPerQuad ; ++ i ) {
    PyList_SET_ITEM( list, i, PyInt_FromLong(data[i]) );
  }

  return list;
}

PyObject*
readOnly( PyObject* self, PyObject* )
{
  Pds::CsPad::ConfigV1QuadReg* obj = pypdsdata::CsPad::ConfigV1QuadReg::pdsObject( self );
  if ( not obj ) return 0;

  return pypdsdata::CsPad::CsPadReadOnlyCfg::PyObject_FromPds( 
      obj->readOnly(), self, sizeof(Pds::CsPad::CsPadReadOnlyCfg) );
}

PyObject*
digitalPots( PyObject* self, PyObject* )
{
  Pds::CsPad::ConfigV1QuadReg* obj = pypdsdata::CsPad::ConfigV1QuadReg::pdsObject( self );
  if ( not obj ) return 0;

  return pypdsdata::CsPad::CsPadDigitalPotsCfg::PyObject_FromPds( 
      &obj->dp(), self, sizeof(Pds::CsPad::CsPadDigitalPotsCfg) );
}

PyObject*
gainMap( PyObject* self, PyObject* )
{
  Pds::CsPad::ConfigV1QuadReg* obj = pypdsdata::CsPad::ConfigV1QuadReg::pdsObject( self );
  if ( not obj ) return 0;

  return pypdsdata::CsPad::CsPadGainMapCfg::PyObject_FromPds( 
      obj->gm(), self, sizeof(Pds::CsPad::CsPadGainMapCfg) );
}

}
