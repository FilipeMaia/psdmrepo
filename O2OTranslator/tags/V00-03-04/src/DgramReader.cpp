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
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/DgramReader.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/DgramQueue.h"
#include "O2OTranslator/O2OXtcMerger.h"
#include "pdsdata/xtc/Dgram.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
DgramReader::DgramReader ( const FileList& files, DgramQueue& queue, size_t maxDgSize, O2OXtcMerger::MergeMode mode )
  : m_files( files )
  , m_queue( queue )
  , m_maxDgSize( maxDgSize )
  , m_mode( mode )
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
{
  O2OXtcMerger iter(m_files, m_maxDgSize, m_mode);
  while ( Pds::Dgram* dg = iter.next() ) {

    // move it to the queue
    m_queue.push ( dg ) ;

  }

  // tell all we are done
  m_queue.push ( (Pds::Dgram*)0 ) ;

}


} // namespace O2OTranslator
