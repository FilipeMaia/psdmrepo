//--------------------------------------------------------------------------
// File and Version Information:
//  $Id: XtcEmbedded.cpp 5295 2013-02-04 20:35:30Z salnikov@SLAC.STANFORD.EDU $
//
// Description:
//  Class XtcEmbedded...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcEmbedded.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff

  char typedoc[] = "Python class wrapping C++ boost::shared_ptr<Pds::Xtc> class.";
}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::XtcEmbedded::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;

  BaseType::initType( "XtcEmbedded", module );
}
