//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnccdFullFrameV1Proxy...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_pds2psana/PnccdFullFrameV1Proxy.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psddl_pds2psana/PnccdFullFrameV1.h"
#include "PSEvt/ProxyDictI.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psddl_pds2psana {

//----------------
// Constructors --
//----------------
PnccdFullFrameV1Proxy::PnccdFullFrameV1Proxy (const boost::shared_ptr<Pds::Xtc>& xtcObj)
  : PSEvt::Proxy<Psana::PNCCD::FullFrameV1>()
  , m_psObj()
{
}

//--------------
// Destructor --
//--------------
PnccdFullFrameV1Proxy::~PnccdFullFrameV1Proxy ()
{
}

boost::shared_ptr<Psana::PNCCD::FullFrameV1>
PnccdFullFrameV1Proxy::getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key)
{
  if (not m_psObj.get()) {
    // get FramesV1 object from event
    boost::shared_ptr<void> vptr = dict->get(&typeid(const Psana::PNCCD::FramesV1), PSEvt::Source(source), key, 0);
    if (vptr) {
      boost::shared_ptr<Psana::PNCCD::FramesV1> frames = boost::static_pointer_cast<Psana::PNCCD::FramesV1>(vptr);
      m_psObj = boost::make_shared<PnccdFullFrameV1>(*frames);
    }
  }
  return m_psObj;
}


} // namespace psddl_pds2psana
