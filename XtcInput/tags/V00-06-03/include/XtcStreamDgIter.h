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
#include <list>
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
#include "XtcInput/Dgram.h"
#include "XtcInput/XtcFileName.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

class XtcChunkDgIter ;

/**
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
   *  Constructor accepts the list of files, the files will be sorted
   *  based on the chunk number extracted from files name.
   *
   *  @param[in]  files    List of input files
   *  @param[in]  maxDgSize Maximum allowed datagram size
   */
  XtcStreamDgIter(const boost::shared_ptr<ChunkFileIterI>& chunkIter, size_t maxDgSize);

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
  Dgram::ptr next() ;

  /// Get file name of currently open chunk
  XtcFileName chunkName() const ;

protected:

private:

  boost::shared_ptr<ChunkFileIterI> m_chunkIter;  ///< Iterator over chunk file names
  size_t m_maxDgSize ;                  ///< Maximum allowed datagram size
  XtcFileName m_file;                   ///< Name of the current chunk
  boost::shared_ptr<XtcChunkDgIter> m_dgiter ;  ///< Datagram iterator for current chunk
  uint64_t m_count ;                    ///< Datagram counter for current chunk

};

} // namespace XtcInput

#endif // XTCINPUT_XTCSTREAMDGITER_H
