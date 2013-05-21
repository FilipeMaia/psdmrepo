//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TdcDataV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TdcDataV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../../EnumType.h"
#include "../TypeLib.h"
#include "TdcDataV1Channel.h"
#include "TdcDataV1Common.h"
#include "TdcDataV1Marker.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace pypdsdata::Acqiris;

namespace {

  pypdsdata::EnumType::Enum sourceEnumValues[] = {
      { "Comm",   Pds::Acqiris::TdcDataV1::Comm },
      { "Chan1",  Pds::Acqiris::TdcDataV1::Chan1 },
      { "Chan2",  Pds::Acqiris::TdcDataV1::Chan2 },
      { "Chan3",  Pds::Acqiris::TdcDataV1::Chan3 },
      { "Chan4",  Pds::Acqiris::TdcDataV1::Chan4 },
      { "Chan5",  Pds::Acqiris::TdcDataV1::Chan5 },
      { "Chan6",  Pds::Acqiris::TdcDataV1::Chan6 },
      { "AuxIO",  Pds::Acqiris::TdcDataV1::AuxIO },
      { 0, 0 }
  };
  pypdsdata::EnumType sourceEnum ( "Source", sourceEnumValues );

  // methods
  PyObject* data( PyObject* self, PyObject* );

  PyMethodDef methods[] = {
    {"data",    data,    METH_NOARGS,
        "self.data() -> list\n\nReturns list of :py:class:`TdcDataV1Channel`, :py:class:`TdcDataV1Common`, and :py:class:`TdcDataV1Marker` objects" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Acqiris::TdcDataV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Acqiris::TdcDataV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums and constants
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "Source", ::sourceEnum.type() );

  BaseType::initType( "TdcDataV1", module );
}

const pypdsdata::EnumType& 
pypdsdata::Acqiris::TdcDataV1::sourceEnum()
{
  return ::sourceEnum;
}

void 
pypdsdata::Acqiris::TdcDataV1::print(std::ostream& out) const
{
  if(not m_obj) return;

  // get the number of items
  const size_t itemSize = sizeof(Pds::Acqiris::TdcDataV1);
  size_t count = m_size / itemSize;

  out << "acqiris.TdcDataV1([" ;

  for (size_t i = 0; i != count; ++ i) {
    if (i) out << ", ";

    switch (m_obj[i].source()) {
    case Pds::Acqiris::TdcDataV1::Comm:
    {
      const Pds::Acqiris::TdcDataV1::Common& item = 
          static_cast<const Pds::Acqiris::TdcDataV1::Common&>(m_obj[i]);
      out << "Common(over=" << item.overflow() << ", nhits=" << item.nhits() << ")";
      break;
    }
    case Pds::Acqiris::TdcDataV1::Chan1:
    case Pds::Acqiris::TdcDataV1::Chan2:
    case Pds::Acqiris::TdcDataV1::Chan3:
    case Pds::Acqiris::TdcDataV1::Chan4:
    case Pds::Acqiris::TdcDataV1::Chan5:
    case Pds::Acqiris::TdcDataV1::Chan6:
    {
      const Pds::Acqiris::TdcDataV1::Channel& item = 
          static_cast<const class Pds::Acqiris::TdcDataV1::Channel&>(m_obj[i]);
      out << "Channel(over=" << item.overflow() << ", ticks=" << item.ticks() << ")";
      break;
    }
    case Pds::Acqiris::TdcDataV1::AuxIO:
    {
      const Pds::Acqiris::TdcDataV1::Marker& item = 
          static_cast<const class Pds::Acqiris::TdcDataV1::Marker&>(m_obj[i]);
      out << "Marker(type=" << item.type() << ")";
      break;
    }
    }
  }

  out << "])" ;
}

namespace {

PyObject*
data( PyObject* self, PyObject* )
{
  const Pds::Acqiris::TdcDataV1* obj = TdcDataV1::pdsObject( self );
  if ( not obj ) return 0;

  const size_t itemSize = sizeof(Pds::Acqiris::TdcDataV1);

  // check that we have whole number of items
  TdcDataV1* py_this = static_cast<TdcDataV1*>(self);
  if ( py_this->m_size % itemSize != 0 ) {
    PyErr_Format(pypdsdata::exceptionType(), "Error: TdcDataV1 XTC object has odd size: %zu", py_this->m_size);
    return 0;
  }
  
  // get the number of items
  size_t count = py_this->m_size / itemSize;

  // make a list
  PyObject* list = PyList_New(count);
  for (size_t i = 0; i != count; ++ i) {
    
    PyObject* pyitem = 0;
    
    switch (obj[i].source()) {
    case Pds::Acqiris::TdcDataV1::Comm:
      pyitem = TdcDataV1Common::PyObject_FromPds((class Pds::Acqiris::TdcDataV1::Common*)(&obj[i]), self, itemSize);
      break;
    case Pds::Acqiris::TdcDataV1::Chan1:
    case Pds::Acqiris::TdcDataV1::Chan2:
    case Pds::Acqiris::TdcDataV1::Chan3:
    case Pds::Acqiris::TdcDataV1::Chan4:
    case Pds::Acqiris::TdcDataV1::Chan5:
    case Pds::Acqiris::TdcDataV1::Chan6:
      pyitem = TdcDataV1Channel::PyObject_FromPds((Pds::Acqiris::TdcDataV1::Channel*)(&obj[i]), self, itemSize);
      break;
    case Pds::Acqiris::TdcDataV1::AuxIO:
      pyitem = TdcDataV1Marker::PyObject_FromPds((Pds::Acqiris::TdcDataV1::Marker*)(&obj[i]), self, itemSize);
      break;
    }
    if (not pyitem) {
      PyErr_Format(pypdsdata::exceptionType(), 
                   "Error: unexpected enum value returned from source(): %d", 
                   int(obj[i].source()));
      return 0;
    }
    PyList_SET_ITEM( list, i, pyitem );
  }
  
  return list;
}

}
