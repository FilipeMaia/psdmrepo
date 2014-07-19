//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DgramSourceWorker...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcMPInput/DgramSourceWorker.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/foreach.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSXtcMPInput/Exceptions.h"
#include "PSXtcMPInput/XtcMPDgramSerializer.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSXtcMPInput {

//----------------
// Constructors --
//----------------
DgramSourceWorker::DgramSourceWorker (const std::string& name)
  : IDatagramSource()
  , psana::Configurable(name)
  , m_fdDataPipe(-1)
  , m_workerId(-1)
  , m_fdReadyPipe(-1)
  , m_ready(false)
{
}

//--------------
// Destructor --
//--------------
DgramSourceWorker::~DgramSourceWorker ()
{
}

// Initialization method for datagram source
void
DgramSourceWorker::init()
{
  // parameters come from module configuration
  m_fdDataPipe = config("fdDataPipe");
  m_workerId = config("workerId");
  m_fdReadyPipe = config("fdReadyPipe");

  // will throw if no files were defined in config
  MsgLog(name(), debug, "worker #" << m_workerId << " reading data from fd " << m_fdDataPipe);

}


//  This method returns next datagram from the source
bool
DgramSourceWorker::next(std::vector<XtcInput::Dgram>& eventDg, std::vector<XtcInput::Dgram>& nonEventDg)
{
  if (not m_ready) {
    // tell master that I'm ready
    uint8_t wid = m_workerId;
    ssize_t sent = ::write(m_fdReadyPipe, &wid, 1);
    if (sent < 0) {
      throw ExceptionErrno(ERR_LOC, "writing to ready pipe failed");
    }
    m_ready = true;
  }
  
  XtcMPDgramSerializer ser(m_fdDataPipe);

  ser.deserialize(eventDg, nonEventDg);

  BOOST_FOREACH (const XtcInput::Dgram& dg, eventDg) {
    // if master sent me L1Accept then it checked my ready flag so I will 
    // need to resend flag on next call
    if (dg.dg()->seq.service() == Pds::TransitionId::L1Accept) {
      m_ready = false;
    }
  }

  return not eventDg.empty();
}

} // namespace PSXtcMPInput
