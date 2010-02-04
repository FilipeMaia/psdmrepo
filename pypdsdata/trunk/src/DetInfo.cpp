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
#include "Level.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  int DetInfo_init( PyObject* self, PyObject* args, PyObject* kwds );
  void DetInfo_dealloc( PyObject* self );
  long DetInfo_hash( PyObject* self );
  int DetInfo_compare( PyObject *self, PyObject *other);
  PyObject* DetInfo_str( PyObject *self );
  PyObject* DetInfo_repr( PyObject *self );

  // type-specific methods
  PyObject* DetInfo_level( PyObject* self, PyObject* );
  PyObject* DetInfo_log( PyObject* self, PyObject* );
  PyObject* DetInfo_phy( PyObject* self, PyObject* );
  PyObject* DetInfo_processId( PyObject* self, PyObject* );
  PyObject* DetInfo_detector( PyObject* self, PyObject* );
  PyObject* DetInfo_device( PyObject* self, PyObject* );
  PyObject* DetInfo_detId( PyObject* self, PyObject* );
  PyObject* DetInfo_devId( PyObject* self, PyObject* );

  PyMethodDef DetInfo_Methods[] = {
    { "level",     DetInfo_level,     METH_NOARGS, "Returns source level object (Level class)" },
    { "log",       DetInfo_log,       METH_NOARGS, "Returns logical address of data source" },
    { "phy",       DetInfo_phy,       METH_NOARGS, "Returns physical address of data source" },
    { "processId", DetInfo_processId, METH_NOARGS, "Returns process ID" },
    { "detector",  DetInfo_detector,  METH_NOARGS, "Returns detector type" },
    { "device",    DetInfo_device,    METH_NOARGS, "Returns device type" },
    { "detId",     DetInfo_detId,     METH_NOARGS, "Returns detector ID" },
    { "devId",     DetInfo_devId,     METH_NOARGS, "Returns device ID" },
    {0, 0, 0, 0}
   };

  char DetInfo_doc[] = "Python class wrapping C++ Pds::DetInfo class.\n\n"
      "Constructor takes five positional arguments, same values ast the\n"
      "C++ constructor. Class implements usual comparison operators\n"
      "and hash function so that objects can be used as dictionary keys.";

  PyTypeObject DetInfo_Type = {
    PyObject_HEAD_INIT(0)
    0,                       /*ob_size*/
    "pdsdata.DetInfo",       /*tp_name*/
    sizeof(pypdsdata::DetInfo), /*tp_basicsize*/
    0,                       /*tp_itemsize*/
    /* methods */
    DetInfo_dealloc,         /*tp_dealloc*/
    0,                       /*tp_print*/
    0,                       /*tp_getattr*/
    0,                       /*tp_setattr*/
    DetInfo_compare,         /*tp_compare*/
    DetInfo_repr,            /*tp_repr*/
    0,                       /*tp_as_number*/
    0,                       /*tp_as_DetInfo*/
    0,                       /*tp_as_mapping*/
    DetInfo_hash,            /*tp_hash*/
    0,                       /*tp_call*/
    DetInfo_str,             /*tp_str*/
    PyObject_GenericGetAttr, /*tp_getattro*/
    PyObject_GenericSetAttr, /*tp_setattro*/
    0,                       /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,      /*tp_flags*/
    DetInfo_doc,             /*tp_doc*/
    0,                       /*tp_traverse*/
    0,                       /*tp_clear*/
    0,                       /*tp_richcompare*/
    0,                       /*tp_weaklistoffset*/
    0,                       /*tp_iter*/
    0,                       /*tp_iternext*/
    DetInfo_Methods,         /*tp_methods*/
    0,                       /*tp_members*/
    0,                       /*tp_getset*/
    0,                       /*tp_base*/
    0,                       /*tp_dict*/
    0,                       /*tp_descr_get*/
    0,                       /*tp_descr_set*/
    0,                       /*tp_dictoffset*/
    DetInfo_init,            /*tp_init*/
    PyType_GenericAlloc,     /*tp_alloc*/
    PyType_GenericNew,       /*tp_new*/
    _PyObject_Del,           /*tp_free*/
    0,                       /*tp_is_gc*/
    0,                       /*tp_bases*/
    0,                       /*tp_mro*/
    0,                       /*tp_cache*/
    0,                       /*tp_subclasses*/
    0,                       /*tp_weaklist*/
    DetInfo_dealloc          /*tp_del*/
  };

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

namespace pypdsdata {


PyTypeObject*
DetInfo::typeObject()
{
  static bool once = true;
  if (once) {
    once = false;

    // define class attributes for enums
    PyObject* tp_dict = PyDict_New();

    PyDict_SetItemString( tp_dict, "NoDetector", PyInt_FromLong(Pds::DetInfo::NoDetector) );
    PyDict_SetItemString( tp_dict, "AmoIms", PyInt_FromLong(Pds::DetInfo::AmoIms) );
    PyDict_SetItemString( tp_dict, "AmoGasdet", PyInt_FromLong(Pds::DetInfo::AmoGasdet) );
    PyDict_SetItemString( tp_dict, "AmoETof", PyInt_FromLong(Pds::DetInfo::AmoETof) );
    PyDict_SetItemString( tp_dict, "AmoITof", PyInt_FromLong(Pds::DetInfo::AmoITof) );
    PyDict_SetItemString( tp_dict, "AmoMbes", PyInt_FromLong(Pds::DetInfo::AmoMbes) );
    PyDict_SetItemString( tp_dict, "AmoVmi", PyInt_FromLong(Pds::DetInfo::AmoVmi) );
    PyDict_SetItemString( tp_dict, "AmoBps", PyInt_FromLong(Pds::DetInfo::AmoBps) );
    PyDict_SetItemString( tp_dict, "Camp", PyInt_FromLong(Pds::DetInfo::Camp) );
    PyDict_SetItemString( tp_dict, "EpicsArch", PyInt_FromLong(Pds::DetInfo::EpicsArch) );
    PyDict_SetItemString( tp_dict, "BldEb", PyInt_FromLong(Pds::DetInfo::BldEb) );
    PyDict_SetItemString( tp_dict, "NumDetector", PyInt_FromLong(Pds::DetInfo::NumDetector) );

    PyDict_SetItemString( tp_dict, "NoDevice", PyInt_FromLong(Pds::DetInfo::NoDevice) );
    PyDict_SetItemString( tp_dict, "Evr", PyInt_FromLong(Pds::DetInfo::Evr) );
    PyDict_SetItemString( tp_dict, "Acqiris", PyInt_FromLong(Pds::DetInfo::Acqiris) );
    PyDict_SetItemString( tp_dict, "Opal1000", PyInt_FromLong(Pds::DetInfo::Opal1000) );
    PyDict_SetItemString( tp_dict, "TM6740", PyInt_FromLong(Pds::DetInfo::TM6740) );
    PyDict_SetItemString( tp_dict, "pnCCD", PyInt_FromLong(Pds::DetInfo::pnCCD) );
    PyDict_SetItemString( tp_dict, "NumDevice", PyInt_FromLong(Pds::DetInfo::NumDevice) );

    DetInfo_Type.tp_dict = tp_dict;
  }

  return &::DetInfo_Type;
}

// makes new DetInfo object from Pds type
PyObject*
DetInfo::DetInfo_FromPds(const Pds::DetInfo& src)
{
  pypdsdata::DetInfo* ob = PyObject_New(pypdsdata::DetInfo,&::DetInfo_Type);
  if ( not ob ) {
    PyErr_SetString( PyExc_RuntimeError, "Failed to create DetInfo object." );
    return 0;
  }

  new(&ob->m_src) Pds::DetInfo(src);

  return (PyObject*)ob;
}

} // namespace pypdsdata

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

  new(&py_this->m_src) Pds::DetInfo(processId, Pds::DetInfo::Detector(det), detId,
      Pds::DetInfo::Device(dev), devId);

  return 0;
}


void
DetInfo_dealloc( PyObject* self )
{
  // deallocate ourself
  self->ob_type->tp_free(self);
}

long
DetInfo_hash( PyObject* self )
{
  pypdsdata::DetInfo* py_this = (pypdsdata::DetInfo*) self;
  int64_t log = py_this->m_src.log() ;
  int64_t phy = py_this->m_src.phy() ;
  long hash = log | ( phy << 32 ) ;
  return hash;
}

int
DetInfo_compare( PyObject* self, PyObject* other )
{
  pypdsdata::DetInfo* py_this = (pypdsdata::DetInfo*) self;
  pypdsdata::DetInfo* py_other = (pypdsdata::DetInfo*) other;
  if ( py_this->m_src.log() > py_other->m_src.log() ) return 1 ;
  if ( py_this->m_src.log() < py_other->m_src.log() ) return -1 ;
  if ( py_this->m_src.phy() > py_other->m_src.phy() ) return 1 ;
  if ( py_this->m_src.phy() < py_other->m_src.phy() ) return -1 ;
  return 0 ;
}

PyObject*
DetInfo_str( PyObject *self )
{
  pypdsdata::DetInfo* py_this = (pypdsdata::DetInfo*) self;
  return PyString_FromString( Pds::DetInfo::name(py_this->m_src) );
}

PyObject*
DetInfo_repr( PyObject *self )
{
  pypdsdata::DetInfo* py_this = (pypdsdata::DetInfo*) self;
  char buf[64];
  snprintf( buf, sizeof buf, "<DetInfo(%d, %s, %d, %s, %d)>",
      py_this->m_src.processId(),
      Pds::DetInfo::name(py_this->m_src.detector()),
      py_this->m_src.detId(),
      Pds::DetInfo::name(py_this->m_src.device()),
      py_this->m_src.devId() );
  return PyString_FromString( buf );
}

PyObject*
DetInfo_level( PyObject* self, PyObject* )
{
  pypdsdata::DetInfo* py_this = (pypdsdata::DetInfo*) self;
  return pypdsdata::Level::Level_FromInt( py_this->m_src.level() );
}

PyObject*
DetInfo_log( PyObject* self, PyObject* )
{
  pypdsdata::DetInfo* py_this = (pypdsdata::DetInfo*) self;
  return PyInt_FromLong( py_this->m_src.log() );
}

PyObject*
DetInfo_phy( PyObject* self, PyObject* )
{
  pypdsdata::DetInfo* py_this = (pypdsdata::DetInfo*) self;
  return PyInt_FromLong( py_this->m_src.phy() );
}

PyObject*
DetInfo_processId( PyObject* self, PyObject* )
{
  pypdsdata::DetInfo* py_this = (pypdsdata::DetInfo*) self;
  return PyInt_FromLong( py_this->m_src.processId() );
}

PyObject*
DetInfo_detector( PyObject* self, PyObject* )
{
  pypdsdata::DetInfo* py_this = (pypdsdata::DetInfo*) self;
  return PyInt_FromLong( py_this->m_src.detector() );
}

PyObject*
DetInfo_device( PyObject* self, PyObject* )
{
  pypdsdata::DetInfo* py_this = (pypdsdata::DetInfo*) self;
  return PyInt_FromLong( py_this->m_src.device() );
}

PyObject*
DetInfo_detId( PyObject* self, PyObject* )
{
  pypdsdata::DetInfo* py_this = (pypdsdata::DetInfo*) self;
  return PyInt_FromLong( py_this->m_src.detId() );
}

PyObject*
DetInfo_devId( PyObject* self, PyObject* )
{
  pypdsdata::DetInfo* py_this = (pypdsdata::DetInfo*) self;
  return PyInt_FromLong( py_this->m_src.devId() );
}

}
