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

  char typedoc[] = "Python class wrapping C++ Pds::BldDataAcqADCV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::BldDataAcqADCV1::initType( PyObject* module )
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
  Pds::BldDataAcqADCV1* pdsObj = pypdsdata::BldDataAcqADCV1::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Acqiris::ConfigV1::PyObject_FromPds(&pdsObj->config,
      self, sizeof(pdsObj->config));
}

PyObject* 
data( PyObject* self, void* )
{
  Pds::BldDataAcqADCV1* pdsObj = pypdsdata::BldDataAcqADCV1::pdsObject(self);
  if(not pdsObj) return 0;

  pypdsdata::BldDataAcqADCV1* pyobj = static_cast<pypdsdata::BldDataAcqADCV1*>(self);
  size_t size = pyobj->m_size - offsetof(Pds::BldDataAcqADCV1, data);
  return pypdsdata::Acqiris::DataDescV1::PyObject_FromPds(&pdsObj->data, self, size);
}

}
