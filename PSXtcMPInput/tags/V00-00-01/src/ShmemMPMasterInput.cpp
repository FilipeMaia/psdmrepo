//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ShmemMPMasterInput...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcMPInput/ShmemMPMasterInput.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSShmemInput/DgramSourceShmem.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace PSXtcMPInput;
PSANA_INPUT_MODULE_FACTORY(ShmemMPMasterInput)


//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

namespace PSXtcMPInput {

//----------------
// Constructors --
//----------------
  ShmemMPMasterInput::ShmemMPMasterInput (const std::string& name)
  : XtcMPMasterInputBase(name, boost::make_shared<PSShmemInput::DgramSourceShmem>(name))
{
}

//--------------
// Destructor --
//--------------
  ShmemMPMasterInput::~ShmemMPMasterInput ()
{
}

} // namespace PSXtcMPInput
