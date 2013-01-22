//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ChunkFileIterList...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/ChunkFileIterList.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

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
ChunkFileIterList::~ChunkFileIterList ()
{
}

/**
 *  @brief Return file name for next chunk.
 *
 *  Returns empty name after the last chunk.
 */
XtcFileName
ChunkFileIterList::next()
{
  XtcFileName next;
  if (not m_chunks.empty()) {
    next = m_chunks.front();
    m_chunks.pop();
  }
  return next;
}

/**
 *  @brief Return live timeout value
 */
unsigned
ChunkFileIterList::liveTimeout() const
{
  return m_liveTimeout;
}

} // namespace XtcInput
