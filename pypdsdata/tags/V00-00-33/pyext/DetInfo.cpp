//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DetInfo...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DetInfo.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <new>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "EnumType.h"
#include "types/TypeLib.h"
#include "Level.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // helper method to get enum name out of enums array
  const char* enum_name(pypdsdata::EnumType::Enum enums[], int val, int maxval)
  {
    // try optimization first based on the order of enums in the list above
    if ( val <= maxval and enums[val].value == val ) {
      return enums[val].name;
    }
    // otherwise try linear search
    for( unsigned i = 0 ; enums[i].name ; ++ i ) {
      if (enums[i].value == val) return enums[i].name;
    }
    return "<Invalid>";    
  }

  pypdsdata::EnumType::Enum detectorEnumValues[] = {
      { "NoDetector",  Pds::DetInfo::NoDetector },
      { "AmoIms",      Pds::DetInfo::AmoIms },
      { "AmoGasdet",   Pds::DetInfo::AmoGasdet },
      { "AmoETof",     Pds::DetInfo::AmoETof },
      { "AmoITof",     Pds::DetInfo::AmoITof },
      { "AmoMbes",     Pds::DetInfo::AmoMbes },
      { "AmoVmi",      Pds::DetInfo::AmoVmi },
      { "AmoBps",      Pds::DetInfo::AmoBps },
      { "Camp",        Pds::DetInfo::Camp },
      { "EpicsArch",   Pds::DetInfo::EpicsArch },
      { "BldEb",       Pds::DetInfo::BldEb },
      { "SxrBeamline", Pds::DetInfo::SxrBeamline },
      { "SxrEndstation", Pds::DetInfo::SxrEndstation },
      { "XppSb1Ipm",   Pds::DetInfo::XppSb1Ipm },
      { "XppSb1Pim",   Pds::DetInfo::XppSb1Pim },
      { "XppMonPim",   Pds::DetInfo::XppMonPim },
      { "XppSb2Ipm",   Pds::DetInfo::XppSb2Ipm },
      { "XppSb3Ipm",   Pds::DetInfo::XppSb3Ipm },
      { "XppSb3Pim",   Pds::DetInfo::XppSb3Pim },
      { "XppSb4Pim",   Pds::DetInfo::XppSb4Pim },
      { "XppGon",      Pds::DetInfo::XppGon },
      { "XppLas",      Pds::DetInfo::XppLas },
      { "XppEndstation", Pds::DetInfo::XppEndstation },
      { "AmoEndstation", Pds::DetInfo::AmoEndstation },
      { "CxiEndstation", Pds::DetInfo::CxiEndstation },
      { "XcsEndstation", Pds::DetInfo::XcsEndstation },
      { "MecEndstation", Pds::DetInfo::MecEndstation },
      { "CxiDg1",      Pds::DetInfo::CxiDg1 },
      { "CxiDg2",      Pds::DetInfo::CxiDg2 },
      { "CxiDg3",      Pds::DetInfo::CxiDg3 },
      { "CxiDg4",      Pds::DetInfo::CxiDg4 },
      { "CxiKb1",      Pds::DetInfo::CxiKb1 },
      { "CxiDs1",      Pds::DetInfo::CxiDs1 },
      { "CxiDs2",      Pds::DetInfo::CxiDs2 },
      { "CxiDsu",      Pds::DetInfo::CxiDsu },
      { "CxiSc1",      Pds::DetInfo::CxiSc1 },
      { "CxiDsd",      Pds::DetInfo::CxiDsd },
      { "NumDetector", Pds::DetInfo::NumDetector },
      { 0, 0 }
  };
  pypdsdata::EnumType detectorEnum ( "Detector", detectorEnumValues );
  inline const char* det_name(Pds::DetInfo::Detector det) {
    return enum_name(detectorEnumValues, int(det), Pds::DetInfo::NumDetector);
  }
  
  const char* det_name(int det) 
  {
    // try optimization first based on the order of enums in the list above
    if ( det <= Pds::DetInfo::NumDetector and detectorEnumValues[det].value == det ) {
      return detectorEnumValues[det].name;
    }
    // otherwise try linear search
    for( unsigned i = 0 ; detectorEnumValues[i].name ; ++ i ) {
      if (detectorEnumValues[i].value == det) return detectorEnumValues[i].name;
    }
    return "<Invalid>";
  }
  
  pypdsdata::EnumType::Enum deviceEnumValues[] = {
      { "NoDevice",  Pds::DetInfo::NoDevice },
      { "Evr",       Pds::DetInfo::Evr },
      { "Acqiris",   Pds::DetInfo::Acqiris },
      { "Opal1000",  Pds::DetInfo::Opal1000 },
      { "TM6740",    Pds::DetInfo::TM6740 },
      { "pnCCD",     Pds::DetInfo::pnCCD },
      { "Princeton", Pds::DetInfo::Princeton },
      { "Fccd",      Pds::DetInfo::Fccd },
      { "Ipimb",     Pds::DetInfo::Ipimb },
      { "Encoder",   Pds::DetInfo::Encoder },
      { "Cspad",     Pds::DetInfo::Cspad },
      { "AcqTDC",    Pds::DetInfo::AcqTDC },
      { "Xamps",     Pds::DetInfo::Xamps },
      { "NumDevice", Pds::DetInfo::NumDevice },
      { 0, 0 }
  };
  pypdsdata::EnumType deviceEnum ( "Device", deviceEnumValues );
  inline const char* dev_name(Pds::DetInfo::Device dev) {
    return enum_name(deviceEnumValues, int(dev), Pds::DetInfo::NumDevice);
  }

  // standard Python stuff
  int DetInfo_init( PyObject* self, PyObject* args, PyObject* kwds );
  long DetInfo_hash( PyObject* self );
  int DetInfo_compare( PyObject *self, PyObject *other);
  PyObject* DetInfo_str( PyObject *self );
  PyObject* DetInfo_repr( PyObject *self );

  // type-specific methods
  PyObject* DetInfo_level( PyObject* self, PyObject* );
  FUN0_WRAPPER_EMBEDDED(pypdsdata::DetInfo, log);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::DetInfo, phy);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::DetInfo, processId);
  ENUM_FUN0_WRAPPER_EMBEDDED(pypdsdata::DetInfo, detector, detectorEnum);
  ENUM_FUN0_WRAPPER_EMBEDDED(pypdsdata::DetInfo, device, deviceEnum);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::DetInfo, detId);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::DetInfo, devId);

  PyMethodDef methods[] = {
    { "level",     DetInfo_level, METH_NOARGS, "Returns source level object (Level class)" },
    { "log",       log,           METH_NOARGS, "Returns logical address of data source" },
    { "phy",       phy,           METH_NOARGS, "Returns physical address of data source" },
    { "processId", processId,     METH_NOARGS, "Returns process ID" },
    { "detector",  detector,      METH_NOARGS, "Returns detector type, one of Detector.AmoIms, Detector.AmoGasdet, etc." },
    { "device",    device,        METH_NOARGS, "Returns device type, one of Device.Evr, Device.Acqiris, etc." },
    { "detId",     detId,         METH_NOARGS, "Returns detector ID" },
    { "devId",     devId,         METH_NOARGS, "Returns device ID" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::DetInfo class.\n\n"
      "Constructor takes five positional arguments, same values ast the\n"
      "C++ constructor. Class implements usual comparison operators\n"
      "and hash function so that objects can be used as dictionary keys.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::DetInfo::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_init = DetInfo_init;
  type->tp_hash = DetInfo_hash;
  type->tp_compare = DetInfo_compare;
  type->tp_str = DetInfo_str;
  type->tp_repr = DetInfo_repr;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Detector", detectorEnum.type() );
  PyDict_SetItemString( tp_dict, "Device", deviceEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "DetInfo", module );
}

namespace {

int
DetInfo_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::DetInfo* py_this = (pypdsdata::DetInfo*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  unsigned processId, det, detId, dev, devId;
  if ( not PyArg_ParseTuple( args, "IIIII:DetInfo", &processId, &det, &detId, &dev, &devId ) ) {
    return -1;
  }

  if ( det >= Pds::DetInfo::NumDetector ) {
    PyErr_SetString(PyExc_TypeError, "Error: detector type out of range");
    return -1;
  }
  if ( dev >= Pds::DetInfo::NumDevice ) {
    PyErr_SetString(PyExc_TypeError, "Error: device type out of range");
    return -1;
  }

  new(&py_this->m_obj) Pds::DetInfo(processId, Pds::DetInfo::Detector(det), detId,
      Pds::DetInfo::Device(dev), devId);

  return 0;
}

long
DetInfo_hash( PyObject* self )
{
  pypdsdata::DetInfo* py_this = (pypdsdata::DetInfo*) self;
  int64_t log = py_this->m_obj.log() ;
  int64_t phy = py_this->m_obj.phy() ;
  long hash = log | ( phy << 32 ) ;
  return hash;
}

int
DetInfo_compare( PyObject* self, PyObject* other )
{
  pypdsdata::DetInfo* py_this = (pypdsdata::DetInfo*) self;
  pypdsdata::DetInfo* py_other = (pypdsdata::DetInfo*) other;
  if ( py_this->m_obj.log() > py_other->m_obj.log() ) return 1 ;
  if ( py_this->m_obj.log() < py_other->m_obj.log() ) return -1 ;
  if ( py_this->m_obj.phy() > py_other->m_obj.phy() ) return 1 ;
  if ( py_this->m_obj.phy() < py_other->m_obj.phy() ) return -1 ;
  return 0 ;
}

PyObject*
DetInfo_str( PyObject *self )
{
  pypdsdata::DetInfo* py_this = (pypdsdata::DetInfo*) self;
  char buf[48];
  snprintf( buf, sizeof buf, "DetInfo(%s-%d|%s-%d)", ::det_name(py_this->m_obj.detector()),
      py_this->m_obj.detId(),
      ::dev_name(py_this->m_obj.device()),
      py_this->m_obj.devId() );
  return PyString_FromString( buf );
}

PyObject*
DetInfo_repr( PyObject *self )
{
  pypdsdata::DetInfo* py_this = (pypdsdata::DetInfo*) self;
  char buf[64];
  snprintf( buf, sizeof buf, "<DetInfo(%d, %s, %d, %s, %d)>",
      py_this->m_obj.processId(),
      ::det_name(py_this->m_obj.detector()),
      py_this->m_obj.detId(),
      ::dev_name(py_this->m_obj.device()),
      py_this->m_obj.devId() );
  return PyString_FromString( buf );
}

PyObject*
DetInfo_level( PyObject* self, PyObject* )
{
  pypdsdata::DetInfo* py_this = (pypdsdata::DetInfo*) self;
  return pypdsdata::Level::Level_FromInt( py_this->m_obj.level() );
}

}
