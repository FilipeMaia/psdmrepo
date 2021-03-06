//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataPimV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataPimV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"
#include "../pulnix/TM6740ConfigV2.h"
#include "../lusi/PimImageConfigV1.h"
#include "../camera/FrameV1.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  PyObject* camConfig( PyObject* self, void* );
  PyObject* pimConfig( PyObject* self, void* );
  PyObject* frame( PyObject* self, void* );
  PyObject* _repr( PyObject *self );

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"camConfig",    camConfig,  0, "attribute of type :py:class:`_pdsdata.pulnix.TM6740ConfigV2`", 0},
    {"pimConfig",    pimConfig,  0, "attribute of type :py:class:`_pdsdata.lusi.PimImageConfigV1`", 0},
    {"frame",        frame,      0, "attribute of type :py:class:`_pdsdata.camera.FrameV1`", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::BldDataPimV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::BldDataPimV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "BldDataPimV1", module );
}

namespace {
  
PyObject* 
camConfig( PyObject* self, void* )
{
  Pds::BldDataPimV1* pdsObj = pypdsdata::BldDataPimV1::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Pulnix::TM6740ConfigV2::PyObject_FromPds(&pdsObj->camConfig,
      self, sizeof(pdsObj->camConfig));
}

PyObject* 
pimConfig( PyObject* self, void* )
{
  Pds::BldDataPimV1* pdsObj = pypdsdata::BldDataPimV1::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Lusi::PimImageConfigV1::PyObject_FromPds(&pdsObj->pimConfig,
      self, sizeof(pdsObj->pimConfig));
}

PyObject* 
frame( PyObject* self, void* )
{
  Pds::BldDataPimV1* pdsObj = pypdsdata::BldDataPimV1::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Camera::FrameV1::PyObject_FromPds(&pdsObj->frame,
      self, sizeof(pdsObj->frame));
}

PyObject*
_repr( PyObject *self )
{
  Pds::BldDataPimV1* pdsObj = pypdsdata::BldDataPimV1::pdsObject(self);
  if(not pdsObj) return 0;

  char buf[96];
  snprintf( buf, sizeof buf, "BldDataPimV1(@%p)", pdsObj );
  return PyString_FromString( buf );
}

}
