//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class SharedFile...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/SharedFile.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "XtcInput/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "XtcInput.SharedFile";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
SharedFile::SharedFileImpl::SharedFileImpl (const XtcFileName& argPath,
    unsigned argLiveTimeout)
  : path(argPath)
  , liveTimeout(argLiveTimeout)
  , fd(-1)
{
  fd = open(path.path().c_str(), O_RDONLY|O_LARGEFILE);
  if (fd < 0) {
    // try to open again after dropping inprogress extension
    if (liveTimeout > 0 and path.extension() == ".inprogress") {
      std::string chop = path.path();
      chop.erase(chop.size()-11);
      path = XtcFileName(chop);
      fd = open(path.path().c_str(), O_RDONLY|O_LARGEFILE);
    }
  }

  // timeout is only used when we read live data
  if (path.extension() != ".inprogress") liveTimeout = 0;

  if (fd < 0) {
    MsgLog( logger, error, "failed to open input XTC file: " << path );
    throw FileOpenException(ERR_LOC, path.path()) ;
  } else {
    MsgLog( logger, trace, "opened input XTC file: " << path << " fd=" << fd);
  }
}

//--------------
// Destructor --
//--------------
SharedFile::SharedFileImpl::~SharedFileImpl()
{
  if (fd >= 0) close(fd);
}


// Read up to size bytes from a file, if EOF is hit
// then check that it is real EOF or wait (in live mode only)
// Returns number of bytes read or negative number for errors.
ssize_t
SharedFile::read(char* buf, size_t size)
{
  std::time_t t0 = m_impl->liveTimeout ? std::time(0) : 0;
  size_t left = size;
  while (left > 0) {
    ssize_t nread = ::read(m_impl->fd, buf+size-left, left);
    MsgLog(logger, debug, "read " << nread << " bytes");
    if (nread == 0) {
      // EOF
      if (t0 and (time(0) - t0) > m_impl->liveTimeout) {
        // in live mode we reached timeout
        MsgLog(logger, error, "Timed out while waiting for data in live mode for file: " << m_impl->path);
        throw XTCLiveTimeout(ERR_LOC, m_impl->path.path(), m_impl->liveTimeout);
      } else if (t0) {
        // in live mode check if we hit real EOF
        if (this->eof()) {
          MsgLog(logger, debug, "Live EOF detected");
          break;
        }
        sleep(1); 
        struct stat buf;
        int f_stat = fstat(m_impl->fd, &buf);
        if ( f_stat == -1 ) {
          MsgLog(logger, warning, "fstat failed for " << m_impl->path);   
        } else {
          MsgLog(logger, debug, "Slept for 1 sec while waiting for more data from file: " << m_impl->path << " size " << buf.st_size);
        }
      } else {
        // non-live mode, just return what we read so far
        break;
      }
    } else if (nread < 0) {
      // error, retry for interrupted reads
      if (errno == EINTR) continue;
      return nread;
    } else {
      // got some data
      left -= nread;
      // reset timeout
      if (t0) t0 = std::time(0);
    }
  }
  return size-left;
}

// check that we reached EOF while reading live data
bool
SharedFile::eof()
{
  // we are at EOF only when the file has been renamed to its final
  // name, but is still the same file (same inode) and it's size is
  // exactly the same as the current offset.

  // strip file extension
  std::string path = m_impl->path.path();
  std::string::size_type p = path.rfind('.');
  if (p == std::string::npos) {
    // file does not have an extension, can't tell anything, just say we
    // are not at the EOF and loop above will timeout eventually
    return false;
  }
  const std::string pathFinal(path, 0, p);

  // check final file, get its info
  struct stat statFinal;
  if (::stat(pathFinal.c_str(), &statFinal) < 0) {
    // no such file, means no EOF yet
    return false;
  }

  // check current position
  off_t offset = ::lseek(m_impl->fd, 0, SEEK_CUR);
  if (offset == (off_t)-1) {
    MsgLog(logger, error, "error returned from lseek: " << errno << " -- " << strerror(errno));
    return false;
  }
  if (offset != statFinal.st_size) {
    // final has size different from our current position
    return false;
  }

  // info for current file
  struct stat statCurrent;
  if (::fstat(m_impl->fd, &statCurrent) < 0) {
    MsgLog(logger, error, "error returned from stat: " << errno << " -- " << strerror(errno));
    return false;
  }

  // compare inodes
  return statFinal.st_dev == statCurrent.st_dev and statFinal.st_ino == statCurrent.st_ino;
}

} // namespace XtcInput
