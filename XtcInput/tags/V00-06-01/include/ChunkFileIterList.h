#ifndef XTCINPUT_CHUNKFILEITERLIST_H
#define XTCINPUT_CHUNKFILEITERLIST_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ChunkFileIterList.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <queue>

//----------------------
// Base Class Headers --
//----------------------
#include "XtcInput/ChunkFileIterI.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

/// @addtogroup XtcInput

/**
 *  @ingroup XtcInput
 *
 *  @brief Implementation of ChunkFileIterI interface based on static list of files.
 *
 *  This implementation is useful for working with non-live data.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class ChunkFileIterList : public ChunkFileIterI {
public:

  /**
   *  @brief Constructor from a sequemce of file names
   *
   *  Constructor takes standard iterators, when dereferenced they should
   *  be convertible to XtcFileName type.
   *
   *  @param[in] begin  Iterator to the beginning of a sequence
   *  @param[in] end    Iterator to the end of a sequence
   *  @param[in] liveTimeout  Timeout for live data, for non-live data should be 0
   */
  template <typename Iter>
  ChunkFileIterList(Iter begin, Iter end, unsigned liveTimeout = 0)
    : ChunkFileIterI()
    , m_chunks()
    , m_liveTimeout(liveTimeout)
  {
    for (; begin != end; ++ begin) {
      m_chunks.push(*begin);
    }
  }

  // Destructor
  virtual ~ChunkFileIterList () ;

  /**
   *  @brief Return file name for next chunk.
   *
   *  Returns empty name after the last chunk.
   */
  virtual XtcFileName next();

  /**
   *  @brief Return live timeout value
   */
  virtual unsigned liveTimeout() const;

protected:

private:

  std::queue<XtcFileName> m_chunks;
  unsigned m_liveTimeout;
  
};

} // namespace XtcInput

#endif // XTCINPUT_CHUNKFILEITERLIST_H
