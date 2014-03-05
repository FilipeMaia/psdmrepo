//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Source...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "Source.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../TypeLib.h"
#include "../../Src.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Partition::Source, group)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Partition::Source, src)
//  long Source_hash( PyObject* self );
//  int Source_compare( PyObject *self, PyObject *other);

  PyMethodDef methods[] = {
    {"group",       group,       METH_NOARGS,  "self.group() -> int\n\nReturns group for src identifier."},
    {"src",         src,         METH_NOARGS,  "self.src() -> object\n\nReturns src identifier (DetInfo, BldInfo, or ProcInfo instance)."},
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Partition::Source class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Partition::Source::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
//  type->tp_hash = ::Source_hash;
//  type->tp_compare = ::Source_compare;

  BaseType::initType( "Source", module );
}

void
pypdsdata::Partition::Source::print(std::ostream& str) const
{
  str << "Source(src=";
  Src::print(str, m_obj.src());
  str << ", group=\"" << m_obj.group() << "\")" ;
}

//namespace {
//
//long Source_hash( PyObject* self )
//{
//  const Pds::Partition::Source& obj = pypdsdata::Partition::Source::pdsObject( self );
//
//  const Pds::Src& src = obj.src();
//  int64_t log = src.log() ;
//  int64_t phy = src.phy() ;
//  long hash = log | ( phy << 32 ) ;
//  return hash;
//}
//
//int Source_compare( PyObject *self, PyObject *other)
//{
//  const Pds::Src& lhs = pypdsdata::Partition::Source::pdsObject(self).src();
//  const Pds::Src& rhs = pypdsdata::Partition::Source::pdsObject(other).src();
//
//  if ( lhs.log() > rhs.log() ) return 1 ;
//  if ( lhs.log() < rhs.log() ) return -1 ;
//  if ( lhs.phy() > rhs.phy() ) return 1 ;
//  if ( lhs.phy() < rhs.phy() ) return -1 ;
//  return 0 ;
//}
//
//}
