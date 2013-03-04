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

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "XtcMergeIterator";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
XtcMergeIterator::XtcMergeIterator (const boost::shared_ptr<RunFileIterI>& runIter, 
    double l1OffsetSec)
  : m_runIter(runIter)
  , m_l1OffsetSec(l1OffsetSec)
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

      // if no more files then stop
      if (not fileNameIter) break ;

      // open next xtc file if there is none open
      MsgLog(logger, trace, "processing run #" << m_runIter->run()) ;
      m_dgiter = boost::make_shared<XtcStreamMerger>(fileNameIter, m_l1OffsetSec);
    }

    // try to read next event from it
    dgram = m_dgiter->next() ;

    // if failed to read go to next file
    if (not dgram.dg()) m_dgiter.reset();

  }

  return dgram ;

}

} // namespace XtcInput
