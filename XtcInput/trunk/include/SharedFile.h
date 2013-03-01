#ifndef XTCINPUT_SHAREDFILE_H
#define XTCINPUT_SHAREDFILE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class SharedFile.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
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
 *  @brief Class representing file that can be shared among several clients.
 *
 *  Class which opens a named file in a constructor and closes it when
 *  the last copy of the object dies. This class does not provide any
 *  thread safety guarantees, use from single thread or protect it yourself.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class SharedFile  {
public:

  /// Default constructor does not open any file, fd() will return negative number
  /// for such files, do not use any other methods on instance created with default constructor.
  SharedFile() {}

  /**
   *  Constructor takes the name of the file.
   *
   *  If liveTimeout is non-zero it means be prepared to read live data.
   *  In this case if file with original name cannot be opened and file
   *  has ".inprogress" extension then it also tries to drop extension
   *  and open file again. If second attempt succeeds then liveTimeout is
   *  reset to 0 (meaning that file is closed and does not need timeouts
   *  when reading).
   */
  SharedFile(const XtcFileName& path, unsigned liveTimeout = 0)
    : m_impl(boost::make_shared<SharedFileImpl>(path, liveTimeout))
  {}

  /// Return file name
  const XtcFileName& path() const { return m_impl->path; }

  /// Return file descriptor
  int fd() const { return m_impl ? m_impl->fd : -1; }

  /// Return timeout value for reading live data
  unsigned liveTimeout() const { return m_impl->liveTimeout; }

  /**
   *  Read up to size bytes from a file, if EOF is hit
   *  then check that it is real EOF or wait (in live mode only)
   *  Returns number of bytes read or negative number for errors.
   */
  ssize_t read(char* buf, size_t size);

  ///  Return information about a file.
  int stat(struct stat *buf) const { return ::fstat(m_impl->fd, buf); }

  ///  Reposition offset of the file, returns new offset.
  off_t seek(off_t offset, int whence) { return ::lseek(m_impl->fd, offset, whence); }


protected:

  // check that we reached EOF while reading live data
  bool eof();

private:

  struct SharedFileImpl {
    SharedFileImpl(const XtcFileName& argPath, unsigned argLiveTimeout);
    ~SharedFileImpl();
    XtcFileName path;
    unsigned liveTimeout;
    int fd;
  };
  
  boost::shared_ptr<SharedFileImpl> m_impl;
};

} // namespace XtcInput

#endif // XTCINPUT_SHAREDFILE_H
