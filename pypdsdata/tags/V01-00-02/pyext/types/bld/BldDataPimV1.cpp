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
#include <stddef.h>

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

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"camConfig",    camConfig,  0, "attribute of type :py:class:`_pdsdata.pulnix.TM6740ConfigV2`", 0},
    {"pimConfig",    pimConfig,  0, "attribute of type :py:class:`_pdsdata.lusi.PimImageConfigV1`", 0},
    {"frame",        frame,      0, "attribute of type :py:class:`_pdsdata.camera.FrameV1`", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Bld::BldDataPimV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Bld::BldDataPimV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "BldDataPimV1", module );
}

namespace {
  
PyObject* 
camConfig( PyObject* self, void* )
{
  Pds::Bld::BldDataPimV1* pdsObj = pypdsdata::Bld::BldDataPimV1::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Pulnix::TM6740ConfigV2::PyObject_FromPds(const_cast<Pds::Pulnix::TM6740ConfigV2*>(&pdsObj->camConfig()),
      self, sizeof(Pds::Pulnix::TM6740ConfigV2));
}

PyObject* 
pimConfig( PyObject* self, void* )
{
  Pds::Bld::BldDataPimV1* pdsObj = pypdsdata::Bld::BldDataPimV1::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Lusi::PimImageConfigV1::PyObject_FromPds(const_cast<Pds::Lusi::PimImageConfigV1*>(&pdsObj->pimConfig()),
      self, sizeof(Pds::Lusi::PimImageConfigV1));
}

PyObject* 
frame( PyObject* self, void* )
{
  Pds::Bld::BldDataPimV1* pdsObj = pypdsdata::Bld::BldDataPimV1::pdsObject(self);
  if(not pdsObj) return 0;

  return pypdsdata::Camera::FrameV1::PyObject_FromPds(const_cast<Pds::Camera::FrameV1*>(&pdsObj->frame()),
      self, pdsObj->frame()._sizeof());
}

}
