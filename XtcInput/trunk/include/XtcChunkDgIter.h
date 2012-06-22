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
   *  @build Create iterator instance
   *
   *  Constructor for iterator which reads datagrams from single XTC file. File must
   *  exist when iterator is instantiated otherwise exception will be thrown. If
   *  live timeout is non-zero then it assumes that file can grow while we are
   *  reading it. It also assumes specific file naming convention, that is live
   *  files have name `dir/basename.ext.extlive` (typically `extlive` is "inprogrees")
   *  and when file is closed it is renamed into `dir/basename.ext`.
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
