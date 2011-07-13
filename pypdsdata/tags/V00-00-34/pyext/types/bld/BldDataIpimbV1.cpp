//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataIpimbV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataIpimbV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"
#include "../ipimb/ConfigV2.h"
#include "../ipimb/DataV2.h"
#include "../lusi/IpmFexV1.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  PyObject* ipimbData( PyObject* self, void* );
  PyObject* ipimbConfig( PyObject* self, void* );
  PyObject* ipmFexData( PyObject* self, void* );
  PyObject* _repr( PyObject *self );

  PyGetSetDef getset[] = {
    {"ipimbData",    ipimbData,    0, "", 0},
    {"ipimbConfig",  ipimbConfig,  0, "", 0},
    {"ipmFexData",   ipmFexData,   0, "", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::BldDataIpimbV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::BldDataIpimbV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "BldDataIpimbV1", module );
}

namespace {
  
PyObject* 
ipimbData( PyObject* self, void* )
{
  Pds::BldDataIpimbV1* pdsObj = pypdsdata::BldDataIpimbV1::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Ipimb::DataV2::PyObject_FromPds(&pdsObj->ipimbData, 
      self, sizeof(pdsObj->ipimbData));
}

PyObject* 
ipimbConfig( PyObject* self, void* )
{
  Pds::BldDataIpimbV1* pdsObj = pypdsdata::BldDataIpimbV1::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Ipimb::ConfigV2::PyObject_FromPds(&pdsObj->ipimbConfig, 
      self, sizeof(pdsObj->ipimbConfig));  
}

PyObject* 
ipmFexData( PyObject* self, void* )
{
  Pds::BldDataIpimbV1* pdsObj = pypdsdata::BldDataIpimbV1::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Lusi::IpmFexV1::PyObject_FromPds(&pdsObj->ipmFexData, 
      self, sizeof(pdsObj->ipmFexData));  
}

PyObject*
_repr( PyObject *self )
{
  Pds::BldDataIpimbV1* pdsObj = pypdsdata::BldDataIpimbV1::pdsObject(self);
  if(not pdsObj) return 0;

  char buf[96];
  snprintf( buf, sizeof buf, "BldDataIpimbV1(@%p)", pdsObj );
  return PyString_FromString( buf );
}

}
