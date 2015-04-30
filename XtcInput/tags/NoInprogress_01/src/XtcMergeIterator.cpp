//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcMergeIterator...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/XtcMergeIterator.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "XtcInput/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "XtcInput.XtcMergeIterator";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
XtcMergeIterator::XtcMergeIterator (const boost::shared_ptr<RunFileIterI>& runIter, 
                                    double l1OffsetSec, int firstControlStream, 
                                    unsigned maxStreamClockDiffSec,
				    boost::shared_ptr<XtcFilesPosition> thirdEvent)
  : m_runIter(runIter)
  , m_l1OffsetSec(l1OffsetSec)
  , m_firstControlStream(firstControlStream)
  , m_maxStreamClockDiffSec(maxStreamClockDiffSec)
  , m_thirdEvent(thirdEvent)
  , m_firstRun(true)

{
}
  
//--------------
// Destructor --
//--------------
XtcMergeIterator::~XtcMergeIterator ()
{
}

// Return next datagram.
Dgram 
XtcMergeIterator::next()
{
  Dgram dgram;
  while (not dgram.dg()) {

    if (not m_dgiter) {

      // get next file name
      boost::shared_ptr<StreamFileIterI> fileNameIter = m_runIter->next();

      boost::shared_ptr<XtcFilesPosition> xtcFilesPos;
      if (m_firstRun) {
	m_firstRun = false;
	if (m_thirdEvent) {
	  if (unsigned(m_thirdEvent->run()) != m_runIter->run()) {
	    MsgLog(logger, error, "run mismatch: thirdEvent.run=" 
		   << m_thirdEvent->run()
		   << " != runIter.run=" << m_runIter->run());
	    throw JumpToDifferentRun(ERR_LOC);
	  }
	  xtcFilesPos = m_thirdEvent;
	}
      }

      // if no more files then stop
      if (not fileNameIter) break ;

      // open next xtc file if there is none open
      MsgLog(logger, trace, "processing run #" << m_runIter->run()) ;
      m_dgiter = boost::make_shared<XtcStreamMerger>(fileNameIter, m_l1OffsetSec, 
                                                     m_firstControlStream,
                                                     m_maxStreamClockDiffSec,
                                                     xtcFilesPos);
    }

    // try to read next datagram from it
    dgram = m_dgiter->next() ;

    // if failed to read go to next file
    if (not dgram.dg()) m_dgiter.reset();

  }

  return dgram ;

}

} // namespace XtcInput
