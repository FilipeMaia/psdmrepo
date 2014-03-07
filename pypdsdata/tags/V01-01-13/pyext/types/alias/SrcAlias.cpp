//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class SrcAlias...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "SrcAlias.h"

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
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Alias::SrcAlias, aliasName)
  PyObject* src( PyObject* self, PyObject* );
  long SrcAlias_hash( PyObject* self );
  int SrcAlias_compare( PyObject *self, PyObject *other);

  PyMethodDef methods[] = {
    {"aliasName",   aliasName,   METH_NOARGS,  "self.aliasName() -> string\n\nReturns alias name for src identifier."},
    {"src",         src,         METH_NOARGS,  "self.src() -> object\n\nReturns src identifier (DetInfo, BldInfo, or ProcInfo instance)."},
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Alias::SrcAlias class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Alias::SrcAlias::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_hash = ::SrcAlias_hash;
  type->tp_compare = ::SrcAlias_compare;

  BaseType::initType( "SrcAlias", module );
}

void
pypdsdata::Alias::SrcAlias::print(std::ostream& str) const
{
  str << "SrcAlias(src=";
  Src::print(str, m_obj.src());
  str << ", alias=\"" << m_obj.aliasName() << "\")" ;
}

namespace {

PyObject* src( PyObject* self, PyObject* )
{
  const Pds::Alias::SrcAlias& obj = pypdsdata::Alias::SrcAlias::pdsObject( self );

  return toPython(obj.src());
}

long SrcAlias_hash( PyObject* self )
{
  const Pds::Alias::SrcAlias& obj = pypdsdata::Alias::SrcAlias::pdsObject( self );

  const Pds::Src& src = obj.src();
  int64_t log = src.log() ;
  int64_t phy = src.phy() ;
  long hash = log | ( phy << 32 ) ;
  return hash;
}

int SrcAlias_compare( PyObject *self, PyObject *other)
{
  const Pds::Src& lhs = pypdsdata::Alias::SrcAlias::pdsObject(self).src();
  const Pds::Src& rhs = pypdsdata::Alias::SrcAlias::pdsObject(other).src();

  if ( lhs.log() > rhs.log() ) return 1 ;
  if ( lhs.log() < rhs.log() ) return -1 ;
  if ( lhs.phy() > rhs.phy() ) return 1 ;
  if ( lhs.phy() < rhs.phy() ) return -1 ;
  return 0 ;
}

}
