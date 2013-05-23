#ifndef XTCINPUT_XTCSTREAMMERGER_H
#define XTCINPUT_XTCSTREAMMERGER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcStreamMerger.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <queue>
#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/Dgram.h"
#include "XtcInput/StreamFileIterI.h"
#include "XtcInput/XtcStreamDgIter.h"
#include "XtcInput/XtcFileName.h"

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
 *  @brief datagram iterator which merges data from several streams.
 *
 *  Class responsible for merging of the datagrams from several
 *  XTC streams.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class XtcStreamMerger : boost::noncopyable {
public:

  /**
   *  @brief Make iterator instance
   *
   *  @param[in]  streamIter  Iterator for input files
   *  @param[in]  l1OffsetSec Time offset to add to non-L1Accept transitions.
   */
  XtcStreamMerger(const boost::shared_ptr<StreamFileIterI>& streamIter,
      double l1OffsetSec = 0 ) ;

  // Destructor
  ~XtcStreamMerger () ;

  /**
   *  @brief Return next datagram.
   *
   *  Read next datagram, return zero pointer after last file has been read,
   *  throws exception for errors.
   *
   *  @return Shared pointer to datagram object
   *
   *  @throw FileOpenException Thrown in case chunk file cannot be open.
   *  @throw XTCReadException Thrown for any read errors
   *  @throw XTCLiveTimeout Thrown for timeout during live data reading
   */
  Dgram next() ;

protected:

  // update time in datagram
  void updateDgramTime(Pds::Dgram& dgram) const ;

private:

  std::vector<boost::shared_ptr<XtcStreamDgIter> > m_streams ;   ///< Set of datagram iterators for individual streams
  std::vector<Dgram> m_dgrams ;               ///< Current datagram for each of the streams
  int32_t m_l1OffsetSec ;                     ///< Time offset to add to non-L1Accept transitions (seconds)
  int32_t m_l1OffsetNsec ;                    ///< Time offset to add to non-L1Accept transitions (nanoseconds)
  std::queue<Dgram> m_outputQueue;            ///< Output queue for datagrams

};

} // namespace XtcInput

#endif // XTCINPUT_XTCSTREAMMERGER_H
