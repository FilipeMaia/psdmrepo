//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcInputModule...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PsXtcInput/XtcInputModule.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <iterator>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace PsXtcInput;
PSANA_INPUT_MODULE_FACTORY(XtcInputModule)

namespace {
  
  const char logger[] = "XtcInputModule";
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PsXtcInput {

//----------------
// Constructors --
//----------------
XtcInputModule::XtcInputModule (const std::string& name)
  : InputModule(name)
{
}

//--------------
// Destructor --
//--------------
XtcInputModule::~XtcInputModule ()
{
}

/// Method which is called once at the beginning of the job
void 
XtcInputModule::beginJob(Env& env)
{
  // will throw if no files were defined in config
  std::list<std::string> files = configList("files");
  WithMsgLog(logger, debug, str) {
    str << "Input files: ";
    std::copy(files.begin(), files.end(), std::ostream_iterator<std::string>(str, " "));
  }
}

InputModule::Status 
XtcInputModule::event(Event& evt, Env& env)
{
  return InputModule::DoEvent;
}

/// Method which is called once at the end of the job
void 
XtcInputModule::endJob(Env& env)
{
}

} // namespace PsXtcInput
