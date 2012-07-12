#ifndef XTCINPUT_STREAMFILEITERLIST_H
#define XTCINPUT_STREAMFILEITERLIST_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class StreamFileIterList.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <queue>
#include <map>
#include <list>

//----------------------
// Base Class Headers --
//----------------------
#include "XtcInput/MergeMode.h"
#include "XtcInput/StreamFileIterI.h"

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
 *  @brief Implementation of StreamFileIterI interface based on static list of files.
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

class StreamFileIterList : public StreamFileIterI {
public:

  // Default constructor
  template <typename Iter>
  StreamFileIterList (Iter begin, Iter end, MergeMode mergeMode, unsigned liveTimeout = 0)
    : StreamFileIterI()
    , m_mergeMode(mergeMode)
    , m_liveTimeout(liveTimeout)
    , m_stream(0)
  {
    switch (m_mergeMode) {
    case MergeFileName:
      for (; begin != end; ++ begin) m_streams[begin->stream()].push_back(*begin);
      break;
    case MergeOneStream:
      for (; begin != end; ++ begin) m_streams[0].push_back(*begin);
      break;
    case MergeNoChunking:
      for (unsigned s = 0; begin != end; ++ begin, ++ s) m_streams[s].push_back(*begin);
      break;
    }
  }

  // Destructor
  virtual ~StreamFileIterList () ;

  /**
   *  @brief Return chunk iterator for next stream.
   *  
   *  Zero pointer is returned after last stream.
   */
  virtual boost::shared_ptr<ChunkFileIterI> next();

  /**
   *  @brief Return stream number for the set of files returned from last next() call.
   */
  virtual unsigned stream() const;

protected:

private:

  typedef std::map<unsigned, std::list<XtcFileName> > Streams;
  
  Streams m_streams;
  MergeMode m_mergeMode;
  unsigned m_liveTimeout;
  unsigned m_stream;

};

} // namespace XtcInput

#endif // XTCINPUT_STREAMFILEITERLIST_H
