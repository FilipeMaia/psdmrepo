//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DgramReader...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/DgramReader.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <boost/make_shared.hpp>
#include <boost/thread/thread.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "XtcInput/DgramQueue.h"
#include "XtcInput/RunFileIterList.h"
#include "XtcInput/XtcMergeIterator.h"
#include "pdsdata/xtc/Dgram.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
DgramReader::DgramReader ( const FileList& files,
                           DgramQueue& queue,
                           size_t maxDgSize,
                           MergeMode mode,
                           bool skipDamaged,
                           double l1OffsetSec )
  : m_files( files )
  , m_queue( queue )
  , m_maxDgSize( maxDgSize )
  , m_mode( mode )
  , m_skipDamaged( skipDamaged )
  , m_l1OffsetSec(l1OffsetSec)
{
}

//--------------
// Destructor --
//--------------
DgramReader::~DgramReader ()
{
}

// this is the "run" method used by the Boost.thread
void
DgramReader::operator() ()
try {

  boost::shared_ptr<RunFileIterI> fileIter =
      boost::make_shared<RunFileIterList>(m_files.begin(), m_files.end(), m_mode);
  XtcMergeIterator iter(fileIter, m_maxDgSize, m_skipDamaged, m_l1OffsetSec);
  Dgram dg;
  while ( not boost::this_thread::interruption_requested() ) {

    dg = iter.next();

    // stop if no datagram
    if (dg.empty()) break;

    // move it to the queue
    m_queue.push ( dg ) ;

  }

  // tell all we are done
  m_queue.push ( Dgram() ) ;

} catch (const boost::thread_interrupted& ex) {

  // we just stop happily
  m_queue.push ( Dgram() ) ;

} catch ( std::exception& e ) {

  MsgLogRoot( error, "exception caught while reading datagram: " << e.what() ) ;
  // TODO: there is no way yet to stop gracefully, will just abort
  throw;

}


} // namespace XtcInput
