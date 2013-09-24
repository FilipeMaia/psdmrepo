//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataAcqADCV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataAcqADCV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <stddef.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"
#include "../acqiris/ConfigV1.h"
#include "../acqiris/DataDescV1.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  PyObject* config( PyObject* self, void* );
  PyObject* data( PyObject* self, void* );

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"config",    config,  0, "attribute of type :py:class:`_pdsdata.acqiris.ConfigV1`", 0},
    {"data",      data,    0, "attribute of type :py:class:`_pdsdata.acqiris.DataDescV1`", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Bld::BldDataAcqADCV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Bld::BldDataAcqADCV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "BldDataAcqADCV1", module );
}

namespace {
  
PyObject* 
config( PyObject* self, void* )
{
  Pds::Bld::BldDataAcqADCV1* pdsObj = pypdsdata::Bld::BldDataAcqADCV1::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Acqiris::ConfigV1::PyObject_FromPds(const_cast<Pds::Acqiris::ConfigV1*>(&pdsObj->config()),
      self, sizeof(Pds::Acqiris::ConfigV1));
}

PyObject* 
data( PyObject* self, void* )
{
  Pds::Bld::BldDataAcqADCV1* pdsObj = pypdsdata::Bld::BldDataAcqADCV1::pdsObject(self);
  if(not pdsObj) return 0;

  pypdsdata::Bld::BldDataAcqADCV1* pyobj = static_cast<pypdsdata::Bld::BldDataAcqADCV1*>(self);
  size_t size = pyobj->m_size - ((const char*)&pdsObj->data() - (const char*)pdsObj);
  return pypdsdata::Acqiris::DataDescV1::PyObject_FromPds(const_cast<Pds::Acqiris::DataDescV1*>(&pdsObj->data()),
      self, size);
}

}
