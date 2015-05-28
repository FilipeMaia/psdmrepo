#ifndef XTCINPUT_XTCSTREAMDGITER_H
#define XTCINPUT_XTCSTREAMDGITER_H

//--------------------------------------------------------------------------
// File and Version Information:
//     $Id$
//
// Description:
//     Class XtcStreamDgIter.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <vector>
#include <cstdio>
#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/ChunkFileIterI.h"
#include "XtcInput/DgHeader.h"
#include "XtcInput/Dgram.h"
#include "XtcInput/XtcFileName.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//             ---------------------
//             -- Class Interface --
//             ---------------------

namespace XtcInput {

class XtcChunkDgIter ;

/// @addtogroup XtcInput

/**
 *  @ingroup XtcInput
 *
 *  @brief Datagram iterator for datagrams in a single XTC stream
 *
 *  Class which merges the chunks from multiple chunks from a single XTC stream.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class XtcStreamDgIter : boost::noncopyable {
public:

  /**
   *  @brief Make iterator instance
   *
   *  Constructor accepts iterator object which iterators over chunks in a stream.
   *
   *  @param[in]  chunkIter Iterator over chunks in a stream
   *  @param[in]  clockSort if we expect out of order L1Accepts, set to true to look for the
   *              correct spot to place a new L1Accept (does not cross non-L1Accept transition
   *              boundaries). Defaults to true. Appropriate for DAQ streams, not neccessary for
   *              Control streams.
   */
  XtcStreamDgIter(const boost::shared_ptr<ChunkFileIterI>& chunkIter,
                  bool clockSort=true);

  /// struct to take a filename and offset for the third datagram in the iteration
  struct ThirdDatagram {
    XtcFileName xtcFile;
    off64_t offset;
    ThirdDatagram() {}
  ThirdDatagram(const XtcFileName &_xtcFile, off64_t _offset) : 
    xtcFile(_xtcFile), offset(_offset) {}
  };

  /**
   *  @brief Make iterator instance that will jump to specified location for second datagram
   *
   *  In addition to taking an iterator object which iterators over chunks in a stream, this
   *  constructor takes a ThirdDatagram struct that gives a chunk file and offset for
   *  the third datagram. After the first and second datagrams in the stream are returned (this is
   *  usually the configure and beginRun transitions) one can effectively jump to an arbitrary 
   *  datagram in the stream - even in another chunk file, as long as the offset of that 
   *  datagram is known.
   *
   *  @param[in] chunkIter as with first constuctor
   *  @param[in] thirdDatagram if non-null, the filename and offset are used for the
   *             third datagram this stream iterator returns. The filename must
   *             exist in the chunkIter or an exception will be thrown from next
   *  @param[in] clockSort as with first constructor
   */
  XtcStreamDgIter(const boost::shared_ptr<ChunkFileIterI>& chunkIter,
                  const boost::shared_ptr<ThirdDatagram> & thirdDatagram,
                  bool clockSort=true);

  // Destructor
  ~XtcStreamDgIter () ;

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

private:

  // fill the read-ahead queue
  void readAhead();

  // add one header to the queue in a correct position
  void queueHeader(const boost::shared_ptr<DgHeader>& header);

  typedef std::vector<boost::shared_ptr<DgHeader> > HeaderQueue;

  boost::shared_ptr<ChunkFileIterI> m_chunkIter;  ///< Iterator over chunk file names
  boost::shared_ptr<XtcChunkDgIter> m_dgiter ;  ///< Datagram iterator for current chunk
  uint64_t m_chunkCount ;                       ///< Datagram counter for current chunk
  uint64_t m_streamCount;                       ///< Datagram counter for stream
  HeaderQueue m_headerQueue;            ///< Queue for read-ahead headers
  bool m_clockSort;                     ///< sort datagrams in between non-L1Accept transitions by clock time
  boost::shared_ptr<ThirdDatagram> m_thirdDatagram;
};

} // namespace XtcInput

#endif // XTCINPUT_XTCSTREAMDGITER_H
