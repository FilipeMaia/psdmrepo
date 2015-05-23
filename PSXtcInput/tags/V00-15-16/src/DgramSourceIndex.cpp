//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: DgramSourceIndex.cpp 7696 2014-02-27 00:40:59Z cpo@SLAC.STANFORD.EDU $
//
// Description:
//	Class DgramSourceIndex...
//
// Author List:
//      Christopher O'Grady
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcInput/DgramSourceIndex.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <queue>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSXtcInput {

//----------------
// Constructors --
//----------------
  DgramSourceIndex::DgramSourceIndex ()
  : IDatagramSource()
{
}

//--------------
// Destructor --
//--------------
DgramSourceIndex::~DgramSourceIndex ()
{}

// Initialization method for datagram source
void 
DgramSourceIndex::init() {
}

//  This method returns next datagram from the source
bool
DgramSourceIndex::next(std::vector<XtcInput::Dgram>& eventDg, std::vector<XtcInput::Dgram>& nonEventDg)
{
  if (not _queue->empty()) {
    DgramPieces pieces = _queue->front();
    eventDg = pieces.eventDg;
    nonEventDg = pieces.nonEventDg;
    _queue->pop();
    return true;
  } else {
    return false;
  }
}

} // namespace PSXtcInput
