#ifndef XTCINPUT_XTCSTREAMDGITER_H
#define XTCINPUT_XTCSTREAMDGITER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcStreamDgIter.
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
namespace XtcInput {
  class FiducialsCompare;
};

//		---------------------
// 		-- Class Interface --
//		---------------------

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
   *  @param[in]  fiducialsCompare, for identying datgrams in the same event
   */
  XtcStreamDgIter(const boost::shared_ptr<ChunkFileIterI>& chunkIter, 
                  const FiducialsCompare &fiducialsCompare);

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
  uint64_t m_count ;                    ///< Datagram counter for current chunk
  HeaderQueue m_headerQueue;            ///< Queue for read-ahead headers
  const FiducialsCompare & m_fiducialsCompare;  ///< for identifying dgrams from same event
};

} // namespace XtcInput

#endif // XTCINPUT_XTCSTREAMDGITER_H
