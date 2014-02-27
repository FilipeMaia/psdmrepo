//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcMPWorkerInput...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcMPInput/XtcMPWorkerInput.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSXtcMPInput/DgramSourceWorker.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace PSXtcMPInput;
PSANA_INPUT_MODULE_FACTORY(XtcMPWorkerInput)


//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

namespace PSXtcMPInput {

//----------------
// Constructors --
//----------------
XtcMPWorkerInput::XtcMPWorkerInput (const std::string& name)
  : PSXtcInput::XtcInputModuleBase(name, boost::make_shared<DgramSourceWorker>(name), true)
{
}

//--------------
// Destructor --
//--------------
XtcMPWorkerInput::~XtcMPWorkerInput()
{
}

} // namespace PSXtcMPInput
