//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcMPMasterInput...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcMPInput/XtcMPMasterInput.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSXtcInput/DgramSourceFile.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace PSXtcMPInput;
PSANA_INPUT_MODULE_FACTORY(XtcMPMasterInput)


//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

namespace PSXtcMPInput {

//----------------
// Constructors --
//----------------
  XtcMPMasterInput::XtcMPMasterInput (const std::string& name)
  : XtcMPMasterInputBase(name, boost::make_shared<PSXtcInput::DgramSourceFile>(name))
{
}

//--------------
// Destructor --
//--------------
  XtcMPMasterInput::~XtcMPMasterInput ()
{
}

} // namespace PSXtcMPInput
