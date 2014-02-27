//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ShmemInputModule...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSShmemInput/ShmemInputModule.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSShmemInput/DgramSourceShmem.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace PSShmemInput;
PSANA_INPUT_MODULE_FACTORY(ShmemInputModule)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSShmemInput {

//----------------
// Constructors --
//----------------
ShmemInputModule::ShmemInputModule(const std::string& name)
  : PSXtcInput::XtcInputModuleBase(name, boost::make_shared<DgramSourceShmem>(name))
{
}

//--------------
// Destructor --
//--------------
ShmemInputModule::~ShmemInputModule ()
{
}

} // namespace PSShmemInput
