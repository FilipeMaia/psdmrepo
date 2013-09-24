//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TdcConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TdcConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"
#include "TdcChannel.h"
#include "TdcAuxIO.h"
#include "TdcVetoIO.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::Acqiris::TdcConfigV1, veto)
  PyObject* channels( PyObject* self, PyObject* args );
  PyObject* auxio( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"channel",  channels, METH_VARARGS,
        "self.channel([idx: int]) -> TdcChannel object(s)\n\nThis method is an alias for channels() method." },
    {"channels", channels, METH_VARARGS,
        "self.channels([idx: int]) -> TdcChannel object(s)\n\nWithout argument returns the list of "
        ":py:class:`TdcChannel` objects,  if argument is given then returns single object for a given index." },
    {"auxio",    auxio,   METH_VARARGS,
        "self.auxio([idx: int]) -> TdcAuxIO object(s)\n\nWithout argument returns the list of "
        ":py:class:`TdcAuxIO` objects,  if argument is given then returns single object for a given index." },
    {"veto",     veto,    METH_NOARGS,  "self.veto() -> TdcVetoIO object\n\nReturns :py:class:`TdcVetoIO` object" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Acqiris::TdcConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Acqiris::TdcConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums and constants
  PyObject* tp_dict = PyDict_New();
  PyObject* val = PyInt_FromLong(Pds::Acqiris::TdcConfigV1::NChannels);
  PyDict_SetItemString( tp_dict, "NChannels", val );
  val = PyInt_FromLong(Pds::Acqiris::TdcConfigV1::NAuxIO);
  PyDict_SetItemString( tp_dict, "NAuxIO", val );
  type->tp_dict = tp_dict;

  BaseType::initType( "TdcConfigV1", module );
}

void 
pypdsdata::Acqiris::TdcConfigV1::print(std::ostream& out) const
{
  if(not m_obj) return;

  out << "acqiris.TdcConfigV1(...)" ;
}

namespace {

PyObject*
channels( PyObject* self, PyObject* args )
{
  const Pds::Acqiris::TdcConfigV1* obj = pypdsdata::Acqiris::TdcConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  unsigned channel = unsigned(-1);
  if ( not PyArg_ParseTuple( args, "|I:Acqiris.TdcConfigV1.channels", &channel ) ) return 0;

  const ndarray<const Pds::Acqiris::TdcChannel, 1>& channels = obj->channels();

  // if argument is missing the return list of objects, otherwise return single object
  using pypdsdata::TypeLib::toPython;
  if (channel == unsigned(-1)) {
    return toPython(channels);
  } else {
    return toPython(channels[channel]);
  }
}

PyObject*
auxio( PyObject* self, PyObject* args )
{
  const Pds::Acqiris::TdcConfigV1* obj = pypdsdata::Acqiris::TdcConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  unsigned index = unsigned(-1);
  if ( not PyArg_ParseTuple( args, "I:Acqiris.TdcConfigV1.auxio", &index ) ) return 0;

  const ndarray<const Pds::Acqiris::TdcAuxIO, 1>& auxio = obj->auxio();

  // if argument is missing the return list of objects, otherwise return single object
  using pypdsdata::TypeLib::toPython;
  if (index == unsigned(-1)) {
    return toPython(auxio);
  } else {
    return toPython(auxio[index]);
  }
}

}
