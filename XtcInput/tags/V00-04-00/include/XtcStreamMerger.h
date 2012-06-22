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
#include <list>
#include <vector>
#include <queue>
#include <iosfwd>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/Dgram.h"
#include "XtcInput/XtcStreamDgIter.h"
#include "XtcInput/XtcFileName.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

/**
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

  /// Merge modes supported by this iterator class
  enum MergeMode {
    OneStream,     ///< All files come from one stream, chunked
    NoChunking,    ///< Single file per stream, no chunking
    FileName       ///< streams and chunks are determined from file names
  } ;
  
  /**
   *  @brief Make merge mode from string
   *  
   *  @throw InvalidMergeMode Thrown if string does not match the names 
   *    of enum constants
   */
  static MergeMode mergeMode(const std::string& str);

  /**
   *  @brief Make iterator instance
   *
   *  @param[in]  files    List of input files
   *  @param[in]  maxDgSize Maximum allowed datagram size
   *  @param[in]  mode      Merge mode
   *  @param[in]  skipDamaged If true then all damaged datagrams will be skipped
   *  @param[in]  l1OffsetSec Time offset to add to non-L1Accept transitions.
   */
  XtcStreamMerger ( const std::list<XtcFileName>& files,
                 size_t maxDgSize,
                 MergeMode mode,
                 bool skipDamaged,
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

  std::vector<XtcStreamDgIter*> m_streams ;   ///< Set of datagram iterators for individual streams
  std::vector<Dgram> m_dgrams ;               ///< Current datagram for each of the streams
  MergeMode m_mode ;                          ///< Merge mode
  int32_t m_l1OffsetSec ;                     ///< Time offset to add to non-L1Accept transitions (seconds)
  int32_t m_l1OffsetNsec ;                    ///< Time offset to add to non-L1Accept transitions (nanoseconds)
  std::queue<Dgram> m_outputQueue;            ///< Output queue for datagrams

};

/// Insertion operator for enum values
std::ostream&
operator<<(std::ostream& out, XtcStreamMerger::MergeMode mode);

} // namespace XtcInput

#endif // XTCINPUT_XTCSTREAMMERGER_H
