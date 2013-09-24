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

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"ipimbData",    ipimbData,    0, "attribute of type :py:class:`_pdsdata.ipimb.DataV1`", 0},
    {"ipimbConfig",  ipimbConfig,  0, "attribute of type :py:class:`_pdsdata.ipimb.ConfigV1`", 0},
    {"ipmFexData",   ipmFexData,   0, "attribute of type :py:class:`_pdsdata.lusi.IpmFexV1`", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Bld::BldDataIpimbV0 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Bld::BldDataIpimbV0::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "BldDataIpimbV0", module );
}

namespace {
  
PyObject* 
ipimbData( PyObject* self, void* )
{
  Pds::Bld::BldDataIpimbV0* pdsObj = pypdsdata::Bld::BldDataIpimbV0::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Ipimb::DataV1::PyObject_FromPds(const_cast<Pds::Ipimb::DataV1*>(&pdsObj->ipimbData()),
      self, sizeof(Pds::Ipimb::DataV1));
}

PyObject* 
ipimbConfig( PyObject* self, void* )
{
  Pds::Bld::BldDataIpimbV0* pdsObj = pypdsdata::Bld::BldDataIpimbV0::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Ipimb::ConfigV1::PyObject_FromPds(const_cast<Pds::Ipimb::ConfigV1*>(&pdsObj->ipimbConfig()),
      self, sizeof(Pds::Ipimb::ConfigV1));
}

PyObject* 
ipmFexData( PyObject* self, void* )
{
  Pds::Bld::BldDataIpimbV0* pdsObj = pypdsdata::Bld::BldDataIpimbV0::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Lusi::IpmFexV1::PyObject_FromPds(const_cast<Pds::Lusi::IpmFexV1*>(&pdsObj->ipmFexData()),
      self, sizeof(Pds::Lusi::IpmFexV1));
}

}
