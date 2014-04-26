//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: XtcIndexInputModule.cpp 7696 2014-02-27 00:40:59Z salnikov@SLAC.STANFORD.EDU $
//
// Description:
//	Class XtcIndexInputModule...
//
// Author List:
//      Christopher O'Grady
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcInput/XtcIndexInputModule.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/make_shared.hpp>
#include <boost/pointer_cast.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSXtcInput/DgramSourceIndex.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace PSXtcInput;
PSANA_INPUT_MODULE_FACTORY(XtcIndexInputModule)


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSXtcInput {

//----------------
// Constructors --
//----------------
XtcIndexInputModule::XtcIndexInputModule (const std::string& name)
  : XtcInputModuleBase(name, boost::make_shared<DgramSourceIndex>())
  , _idx(name,_queue)
{
  // this is ugly. necessary because it's hard to remember the
  // DgramSourceIndex that we give to XtcInputModuleBase.
  boost::shared_ptr<DgramSourceIndex> dgSourceIndex = boost::dynamic_pointer_cast<DgramSourceIndex>(m_dgsource);
  dgSourceIndex->setQueue(_queue);
}

//--------------
// Destructor --
//--------------
XtcIndexInputModule::~XtcIndexInputModule ()
{
}

} // namespace PSXtcInput
