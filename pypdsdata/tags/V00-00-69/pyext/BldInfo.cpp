//--------------------------------------------------------------------------
// File and Version Information:
//  $Id$
//
// Description:
//  Class BldInfo...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldInfo.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <new>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "EnumType.h"
#include "Level.h"
#include "types/TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

    pypdsdata::EnumType::Enum typeEnumValues[] = {
        { "EBeam",           Pds::BldInfo::EBeam },
        { "PhaseCavity",     Pds::BldInfo::PhaseCavity },
        { "FEEGasDetEnergy", Pds::BldInfo::FEEGasDetEnergy },
        { "Nh2Sb1Ipm01",     Pds::BldInfo::Nh2Sb1Ipm01 },
        { "HxxUm6Imb01",     Pds::BldInfo::HxxUm6Imb01 },
        { "HxxUm6Imb02",     Pds::BldInfo::HxxUm6Imb02 },
        { "HfxDg2Imb01",     Pds::BldInfo::HfxDg2Imb01 },
        { "HfxDg2Imb02",     Pds::BldInfo::HfxDg2Imb02 },
        { "XcsDg3Imb03",     Pds::BldInfo::XcsDg3Imb03 },
        { "XcsDg3Imb04",     Pds::BldInfo::XcsDg3Imb04 },
        { "HfxDg3Imb01",     Pds::BldInfo::HfxDg3Imb01 },
        { "HfxDg3Imb02",     Pds::BldInfo::HfxDg3Imb02 },
        { "HxxDg1Cam",       Pds::BldInfo::HxxDg1Cam },
        { "HfxDg2Cam",       Pds::BldInfo::HfxDg2Cam },
        { "HfxDg3Cam",       Pds::BldInfo::HfxDg3Cam },
        { "XcsDg3Cam",       Pds::BldInfo::XcsDg3Cam },
        { "HfxMonCam",       Pds::BldInfo::HfxMonCam },
        { "HfxMonImb01",     Pds::BldInfo::HfxMonImb01 },
        { "HfxMonImb02",     Pds::BldInfo::HfxMonImb02 },
        { "HfxMonImb03",     Pds::BldInfo::HfxMonImb03 },
        { "MecLasEm01",      Pds::BldInfo::MecLasEm01 },
        { "MecTctrPip01",    Pds::BldInfo::MecTctrPip01 },
        { "MecTcTrDio01",    Pds::BldInfo::MecTcTrDio01 },
        { "MecXt2Ipm02",     Pds::BldInfo::MecXt2Ipm02 },
        { "MecXt2Ipm03",     Pds::BldInfo::MecXt2Ipm03 },
        { "MecHxmIpm01",     Pds::BldInfo::MecHxmIpm01 },
        { "NumberOf",        Pds::BldInfo::NumberOf },
        { 0, 0 }
    };
    pypdsdata::EnumType typeEnum ( "Type", typeEnumValues );


  // standard Python stuff
  int BldInfo_init(PyObject* self, PyObject* args, PyObject* kwds);
  long BldInfo_hash( PyObject* self );
  int BldInfo_compare( PyObject *self, PyObject *other);
  PyObject* BldInfo_str( PyObject *self );
  PyObject* BldInfo_repr( PyObject *self );

  // type-specific methods
  PyObject* BldInfo_level( PyObject* self, PyObject* args );
  FUN0_WRAPPER_EMBEDDED(pypdsdata::BldInfo, log);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::BldInfo, phy);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::BldInfo, processId);
  ENUM_FUN0_WRAPPER_EMBEDDED(pypdsdata::BldInfo, type, typeEnum);

  PyMethodDef methods[] = {
    { "level",     BldInfo_level,     METH_NOARGS, "self.level() -> Level\n\nReturns source level object (:py:class:`Level` class)" },
    { "log",       log,       METH_NOARGS, "self.log() -> int\n\nReturns logical address of data source" },
    { "phy",       phy,       METH_NOARGS, "self.phy() -> int\n\nReturns physical address of data source" },
    { "processId", processId, METH_NOARGS, "self.processId() -> int\n\nReturns process ID" },
    { "type",      type,      METH_NOARGS, "self.type() -> Type\n\nReturns BldInfo :py:class:`Type` enum" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::BldInfo class.\n\n"
      "Constructor takes two positional arguments, same values as the\n"
      "C++ constructor. Class implements usual comparison operators\n"
      "and hash function so that objects can be used as dictionary keys.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::BldInfo::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_init = BldInfo_init;
  type->tp_hash = BldInfo_hash;
  type->tp_compare = BldInfo_compare;
  type->tp_str = BldInfo_str;
  type->tp_repr = BldInfo_repr;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Type", typeEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "BldInfo", module );
}

namespace {

int
BldInfo_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  int processId=0, type=-1;
  if (PyTuple_GET_SIZE(args) == 1) {

    // argument can be an integer (long) or string
    PyObject* obj;
    if (not PyArg_ParseTuple(args, "O:BldInfo", &obj)) return -1;
    if (PyInt_Check(obj)) {
      type = PyInt_AsLong(obj);
    } else if (PyLong_Check(obj)) {
      type = PyLong_AsLong(obj);
    } else if (PyString_Check(obj)) {

      // string may be one of the strings returned from name()
      char* str = PyString_AsString(obj);
      for (int i = 0; i < Pds::BldInfo::NumberOf; ++ i) {
        Pds::BldInfo info(0, Pds::BldInfo::Type(i));
        if (strcmp(str, Pds::BldInfo::name(info)) == 0) {
          type = i;
          break;
        }
      }
      if (type < 0) {
        PyErr_Format(PyExc_ValueError, "Error: unknown BLD type string: %s", str);
        return -1;
      }
    }

  } else {

    // expect two integer arguments
    if (not PyArg_ParseTuple(args, "II:BldInfo", &processId, &type)) return -1;

  }

  if ( type < 0 or type >= Pds::BldInfo::NumberOf ) {
    PyErr_SetString(PyExc_ValueError, "Error: BLD type out of range");
    return -1;
  }

  new(&py_this->m_obj) Pds::BldInfo( processId, Pds::BldInfo::Type(type) );

  return 0;
}

long
BldInfo_hash( PyObject* self )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  int64_t log = py_this->m_obj.log() ;
  int64_t phy = py_this->m_obj.phy() ;
  long hash = log | ( phy << 32 ) ;
  return hash;
}

int
BldInfo_compare( PyObject* self, PyObject* other )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  pypdsdata::BldInfo* py_other = (pypdsdata::BldInfo*) other;
  // only compare Level and Type, skip process IDs
  if ( py_this->m_obj.level() > py_other->m_obj.level() ) return 1 ;
  if ( py_this->m_obj.level() < py_other->m_obj.level() ) return -1 ;
  if ( py_this->m_obj.type() > py_other->m_obj.type() ) return 1 ;
  if ( py_this->m_obj.type() < py_other->m_obj.type() ) return -1 ;
  return 0 ;
}

PyObject*
BldInfo_str( PyObject *self )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  if (py_this->m_obj.type() < Pds::BldInfo::NumberOf) {
    return PyString_FromFormat( "BldInfo(%s)", Pds::BldInfo::name(py_this->m_obj) );
  } else {
    return PyString_FromFormat( "BldInfo(%d)", int(py_this->m_obj.type()) );
  }
}

PyObject*
BldInfo_repr( PyObject *self )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  return PyString_FromFormat( "<BldInfo(%d, %s)>",
      int(py_this->m_obj.processId()),
      Pds::BldInfo::name(py_this->m_obj) );
}

PyObject*
BldInfo_level( PyObject* self, PyObject* )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  return pypdsdata::Level::Level_FromInt( py_this->m_obj.level() );
}

}
