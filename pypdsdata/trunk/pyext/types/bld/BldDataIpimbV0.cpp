//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: BldDataIpimbV0.cpp 811 2010-03-26 17:40:08Z salnikov $
//
// Description:
//	Class BldDataIpimbV0...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataIpimbV0.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"
#include "../ipimb/ConfigV1.h"
#include "../ipimb/DataV1.h"
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

  char typedoc[] = "Python class wrapping C++ Pds::BldDataIpimbV0 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::BldDataIpimbV0::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "BldDataIpimbV0", module );
}

namespace {
  
PyObject* 
ipimbData( PyObject* self, void* )
{
  Pds::BldDataIpimbV0* pdsObj = pypdsdata::BldDataIpimbV0::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Ipimb::DataV1::PyObject_FromPds(&pdsObj->ipimbData, 
      self, sizeof(pdsObj->ipimbData));
}

PyObject* 
ipimbConfig( PyObject* self, void* )
{
  Pds::BldDataIpimbV0* pdsObj = pypdsdata::BldDataIpimbV0::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Ipimb::ConfigV1::PyObject_FromPds(&pdsObj->ipimbConfig, 
      self, sizeof(pdsObj->ipimbConfig));  
}

PyObject* 
ipmFexData( PyObject* self, void* )
{
  Pds::BldDataIpimbV0* pdsObj = pypdsdata::BldDataIpimbV0::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Lusi::IpmFexV1::PyObject_FromPds(&pdsObj->ipmFexData, 
      self, sizeof(pdsObj->ipmFexData));  
}

PyObject*
_repr( PyObject *self )
{
  Pds::BldDataIpimbV0* pdsObj = pypdsdata::BldDataIpimbV0::pdsObject(self);
  if(not pdsObj) return 0;

  char buf[96];
  snprintf( buf, sizeof buf, "BldDataIpimbV0(@%p)", pdsObj );
  return PyString_FromString( buf );
}

}
