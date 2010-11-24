//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: BldDataIpimb.cpp 811 2010-03-26 17:40:08Z salnikov $
//
// Description:
//	Class BldDataIpimb...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataIpimb.h"

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

  char typedoc[] = "Python class wrapping C++ Pds::BldDataIpimb class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::BldDataIpimb::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "BldDataIpimb", module );
}

namespace {
  
PyObject* 
ipimbData( PyObject* self, void* )
{
  Pds::BldDataIpimb* pdsObj = pypdsdata::BldDataIpimb::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Ipimb::DataV1::PyObject_FromPds(&pdsObj->ipimbData, 
      self, sizeof(pdsObj->ipimbData));
}

PyObject* 
ipimbConfig( PyObject* self, void* )
{
  Pds::BldDataIpimb* pdsObj = pypdsdata::BldDataIpimb::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Ipimb::ConfigV1::PyObject_FromPds(&pdsObj->ipimbConfig, 
      self, sizeof(pdsObj->ipimbConfig));  
}

PyObject* 
ipmFexData( PyObject* self, void* )
{
  Pds::BldDataIpimb* pdsObj = pypdsdata::BldDataIpimb::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Lusi::IpmFexV1::PyObject_FromPds(&pdsObj->ipmFexData, 
      self, sizeof(pdsObj->ipmFexData));  
}

PyObject*
_repr( PyObject *self )
{
  Pds::BldDataIpimb* pdsObj = pypdsdata::BldDataIpimb::pdsObject(self);
  if(not pdsObj) return 0;

  char buf[96];
  snprintf( buf, sizeof buf, "BldDataIpimb(@%p)", pdsObj );
  return PyString_FromString( buf );
}

}
