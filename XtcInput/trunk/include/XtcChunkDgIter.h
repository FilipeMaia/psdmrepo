#ifndef XTCINPUT_XTCCHUNKDGITER_H
#define XTCINPUT_XTCCHUNKDGITER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcChunkDgIter.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "XtcInput/Dgram.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

/**
 *  @brief Datagram iterator for datagrams in a single chunk file.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class XtcChunkDgIter : boost::noncopyable {
public:

  /**
   *  @brief Create iterator instance
   *
   *  Constructor for iterator which reads datagrams from single XTC file.
   *  It tries to open specified path first, if it fails and path has
   *  extension ".inprogress" then it removes extension and tries to
   *  open file again. If that fails the exception will be thrown.
   *  If file has an extension ".inprogress" and live timeout is non-zero
   *  then it assumes that file can grow while we are reading it.
   *
   *  @param[in]  path         Path name for XTC file
   *  @param[in]  maxDgramSize Maximum datagram size, if datagram in file exceeds this size
   *                     then an exception will be generated
   *  @param[in]  liveTimeout  If non-zero then defines timeout in seconds for reading live
   *                     data files, if zero assumes that files is closed already
   *
   *  @throw FileOpenException Thrown in case file cannot be open.
   */
  XtcChunkDgIter (const std::string& path, size_t maxDgramSize, unsigned liveTimeout = 0) ;

  // Destructor
  ~XtcChunkDgIter () ;

  /**
   *  @brief Returns next datagram, zero on EOF, throws exceptions for errors
   *
   *  If reading ".inprogress" file stops at EOF and there is no
   *  new data for "live timeout" seconds then XTCLiveTimeout exception
   *  is generated.
   *  File is assumed to be closed and does not grow any more when
   *  it is renamed and ".inprogess" extension is dropped.
   *
   *  @return Shared pointer to datagram object
   *
   *  @throw XTCReadException Thrown for any read errors
   *  @throw XTCLiveTimeout Thrown for timeout during live data reading
   */
  Dgram::ptr next() ;

protected:

  // Read up to size bytes from a file, if EOF is hit
  // then check that it is real EOF or wait (in live mode only)
  // Returns number of bytes read or negative number for errors.
  ssize_t read(char* buf, size_t size);

  // check that we reached EOF while reading live data
  bool eof();

private:

  std::string m_path;      ///< Name of the chunk file
  int m_fd;                ///< File descriptor of the open file
  size_t m_maxDgramSize ;  ///< Maximum allowed datagram size
  unsigned m_liveTimeout;  ///< timeout in seconds for reading live data files

};

} // namespace XtcInput

#endif // XTCINPUT_XTCCHUNKDGITER_H
