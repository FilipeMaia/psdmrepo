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

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
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
   *  @param[in]  skipDamaged If true then all damaged datagrams will be skipped
   */
  XtcStreamDgIter ( const std::list<XtcFileName>& files, size_t maxDgSize, bool skipDamaged ) ;

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

  std::list<XtcFileName> m_files ;      ///< List of input files, ordered by chunk number
  size_t m_maxDgSize ;                  ///< Maximum allowed datagram size
  bool m_skipDamaged ;                  ///< If true then skip damaged datagrams altogether
  std::list<XtcFileName>::const_iterator m_iter ;  ///< Iterator inside m_files list
  XtcChunkDgIter* m_dgiter ;            ///< Datagram iterator for current chunk
  uint64_t m_count ;                    ///< Datagram counter for current chunk

};

} // namespace XtcInput

#endif // XTCINPUT_XTCSTREAMDGITER_H
