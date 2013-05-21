//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PvConfigV1.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::Epics::ConfigV1, getNumPv)
  PyObject* getPvConfigs( PyObject* self, PyObject* );
  PyObject* getPvConfig( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"getNumPv",     getNumPv,      METH_NOARGS,  "self.getNumPv() -> int\n\nReturns number of :py:class:`PvConfigV1` object." },
    {"getPvConfigs", getPvConfigs,  METH_NOARGS,  "self.getPvConfigs() -> list of PvConfigV1\n\nReturns list of :py:class:`PvConfigV1` objects." },
    {"getPvConfig",  getPvConfig,   METH_VARARGS, "self.getPvConfig(idx: int) -> PvConfigV1\n\nReturns :py:class:`PvConfigV1` for a given index." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Epics::ConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Epics::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV1", module );
}

void
pypdsdata::Epics::ConfigV1::print(std::ostream& str) const
{
  str << "epics.ConfigV1(numPv=" << m_obj->getNumPv() << ", PvConfigs=[";
  unsigned size = m_obj->getNumPv();
  for ( unsigned i = 0; i < size and i < 256; ++ i ) {
    const Pds::Epics::PvConfigV1* pv = m_obj->getPvConfig(i);
    if (i) str << ", ";
    str << "(pvId=" << pv->iPvId << ", desc=\"" << pv->sPvDesc
        << "\", interval=" << pv->fInterval << ")";
  }
  if (size > 256) str << ", ...";
  str << "])";
}


namespace {

PyObject*
getPvConfigs( PyObject* self, PyObject* )
{
  Pds::Epics::ConfigV1* obj = pypdsdata::Epics::ConfigV1::pdsObject(self);
  if(not obj) return 0;

  unsigned size = obj->getNumPv();
  PyObject* list = PyList_New(size);

  // copy PvConfig objects to the list
  for ( unsigned i = 0; i < size; ++ i ) {
    PyObject* pvcfg = pypdsdata::Epics::PvConfigV1::PyObject_FromPds(obj->getPvConfig(i),
        self, sizeof(Pds::Epics::PvConfigV1) );
    PyList_SET_ITEM( list, i, pvcfg );
  }

  return list;
}

PyObject*
getPvConfig( PyObject* self, PyObject* args )
{
  Pds::Epics::ConfigV1* obj = pypdsdata::Epics::ConfigV1::pdsObject(self);
  if ( not obj ) return 0;

  // parse args
  unsigned index ;
  if ( not PyArg_ParseTuple( args, "I:ConfigV1_getPvConfig", &index ) ) return 0;

  return pypdsdata::Epics::PvConfigV1::PyObject_FromPds(obj->getPvConfig(index),
      self, sizeof(Pds::Epics::PvConfigV1) );
}

}
