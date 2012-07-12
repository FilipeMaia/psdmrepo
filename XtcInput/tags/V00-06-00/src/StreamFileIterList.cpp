//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class StreamFileIterList...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/StreamFileIterList.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/ChunkFileIterList.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//--------------
// Destructor --
//--------------
StreamFileIterList::~StreamFileIterList ()
{
}

/**
 *  @brief Return chunk iterator for next stream.
 *  
 *  Zero pointer is returned after last stream.
 */
boost::shared_ptr<ChunkFileIterI> 
StreamFileIterList::next()
{
  boost::shared_ptr<ChunkFileIterI> next;
  if (not m_streams.empty()) {
    Streams::iterator s = m_streams.begin();
    m_stream = s->first;
    if (m_mergeMode == MergeFileName) s->second.sort();
    next = boost::make_shared<ChunkFileIterList>(s->second.begin(), s->second.end(), m_liveTimeout);
    m_streams.erase(s);
  }
  
  return next;  
}

/**
 *  @brief Return stream number for the set of files returned from last next() call.
 */
unsigned 
StreamFileIterList::stream() const
{
  return m_stream;
}


} // namespace XtcInput
