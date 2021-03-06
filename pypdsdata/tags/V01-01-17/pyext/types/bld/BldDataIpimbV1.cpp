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

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"ipimbData",    ipimbData,    0, "attribute of type :py:class:`_pdsdata.ipimb.DataV2`", 0},
    {"ipimbConfig",  ipimbConfig,  0, "attribute of type :py:class:`_pdsdata.ipimb.ConfigV2`", 0},
    {"ipmFexData",   ipmFexData,   0, "attribute of type :py:class:`_pdsdata.lusi.IpmFexV1`", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Bld::BldDataIpimbV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Bld::BldDataIpimbV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "BldDataIpimbV1", module );
}

namespace {
  
PyObject* 
ipimbData( PyObject* self, void* )
{
  Pds::Bld::BldDataIpimbV1* pdsObj = pypdsdata::Bld::BldDataIpimbV1::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Ipimb::DataV2::PyObject_FromPds(const_cast<Pds::Ipimb::DataV2*>(&pdsObj->ipimbData()),
      self, sizeof(Pds::Ipimb::DataV2));
}

PyObject* 
ipimbConfig( PyObject* self, void* )
{
  Pds::Bld::BldDataIpimbV1* pdsObj = pypdsdata::Bld::BldDataIpimbV1::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Ipimb::ConfigV2::PyObject_FromPds(const_cast<Pds::Ipimb::ConfigV2*>(&pdsObj->ipimbConfig()),
      self, sizeof(Pds::Ipimb::ConfigV2));
}

PyObject* 
ipmFexData( PyObject* self, void* )
{
  Pds::Bld::BldDataIpimbV1* pdsObj = pypdsdata::Bld::BldDataIpimbV1::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Lusi::IpmFexV1::PyObject_FromPds(const_cast<Pds::Lusi::IpmFexV1*>(&pdsObj->ipmFexData()),
      self, sizeof(Pds::Lusi::IpmFexV1));
}

}
