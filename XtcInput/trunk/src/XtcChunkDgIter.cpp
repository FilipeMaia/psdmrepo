//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcChunkDgIter...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/XtcChunkDgIter.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <ctime>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <boost/filesystem.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/Exceptions.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace fs = boost::filesystem;

namespace {

  const char* logger = "XtcChunkDgIter";

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
XtcChunkDgIter::XtcChunkDgIter (const std::string& path, size_t maxDgramSize, unsigned liveTimeout)
  : m_path(path)
  , m_fd(-1)
  , m_maxDgramSize(maxDgramSize)
  , m_liveTimeout(liveTimeout)
{
  fs::path fpath = m_path;
  m_fd = open(fpath.string().c_str(), O_RDONLY|O_LARGEFILE);
  if (m_fd < 0) {
    // try to open after dropping inprogress extension
    if (fpath.extension() == ".inprogress") {
      fpath.replace_extension();
      m_fd = open(fpath.string().c_str(), O_RDONLY|O_LARGEFILE);
      if (m_fd >= 0) m_path = fpath.string();
    }
  }

  if (m_fd < 0) {
    MsgLog( logger, error, "failed to open input XTC file: " << m_path );
    throw FileOpenException(ERR_LOC, m_path) ;
  } else {
    MsgLog( logger, trace, "opened input XTC file: " << m_path );
  }

  // if reading closed file timeout must be set to 0
  if (fpath.extension() != ".inprogress") {
    m_liveTimeout = 0;
  }

}

//--------------
// Destructor --
//--------------
XtcChunkDgIter::~XtcChunkDgIter ()
{
  if (m_fd >= 0) close(m_fd);
}

Dgram::ptr
XtcChunkDgIter::next()
{
  Pds::Dgram header;
  const size_t headerSize = sizeof(Pds::Dgram);

  // read header
  MsgLog(logger, debug, "reading header");
  ssize_t nread = this->read(((char*)&header), headerSize);
  if (nread == 0) {
    return Dgram::ptr();
  } else if (nread < 0) {
    throw XTCReadException(ERR_LOC, m_path);
  } else if (nread != ssize_t(headerSize)) {
    MsgLog(logger, error, "EOF while reading datagram header from file: " << m_path);
    return Dgram::ptr();
  }

  WithMsgLog(logger, debug, str) {
    str << "header:";
    uint32_t* p = (uint32_t*)&header;
    for (int i = 0; i != 10; ++ i) str << ' ' << p[i];
  }

  // check payload size, protection against corrupted headers
  uint32_t payloadSize = header.xtc.sizeofPayload();
  MsgLog(logger, debug, "payload size = " << payloadSize);
  if (payloadSize > (m_maxDgramSize-headerSize)) {
    throw XTCSizeLimitException(ERR_LOC, m_path, payloadSize+headerSize, m_maxDgramSize);
  }

  Pds::Dgram* dg = (Pds::Dgram*)new char[payloadSize+headerSize];
  std::copy((const char*)&header, ((const char*)&header)+headerSize, (char*)dg);

  // read rest of the data
  MsgLog(logger, debug, "reading payload, size = " << payloadSize);
  nread = this-> read(dg->xtc.payload(), payloadSize);
  if (nread < 0) {
    throw XTCReadException(ERR_LOC, m_path);
  } else if (nread != ssize_t(payloadSize)) {
    MsgLog(logger, error, "EOF while reading datagram payload from file: " << m_path);
    return Dgram::ptr();
  }

  return Dgram::make_ptr(dg);
}

// Read up to size bytes from a file, if EOF is hit
// then check that it is real EOF or wait (in live mode only)
// Returns number of bytes read or negative number for errors.
ssize_t
XtcChunkDgIter::read(char* buf, size_t size)
{
  std::time_t t0 = m_liveTimeout ? std::time(0) : 0;
  size_t left = size;
  while (left > 0) {
    ssize_t nread = ::read(m_fd, buf+size-left, left);
    MsgLog(logger, debug, "read " << nread << " bytes");
    if (nread == 0) {
      // EOF
      if (t0 and (time(0) - t0) > m_liveTimeout) {
        // in live mode we reached timeout
        MsgLog(logger, error, "Timed out while waiting for data in live mode for file: " << m_path);
        throw XTCLiveTimeout(ERR_LOC, m_path, m_liveTimeout);
      } else if (t0) {
        // in live mode check if we hit real EOF
        if (this->eof()) {
          MsgLog(logger, debug, "Live EOF detected");
          break;
        }
        MsgLog(logger, debug, "Sleep for 1 sec while waiting for more data from file: " << m_path);
        sleep(1);
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
XtcChunkDgIter::eof()
{
  // we are at EOF only when the file has been renamed to its final
  // name, but is still the same file (same inode) and it's size is
  // exactly the same as the current offset.

  // strip file extension
  std::string::size_type p = m_path.rfind('.');
  if (p == std::string::npos) {
    // file does not have an extension, can't tell anything, just say we
    // are not at the EOF and loop above will timeout eventually
    return false;
  }
  const std::string pathFinal(m_path, 0, p);

  // check final file, get its info
  struct stat statFinal;
  if (::stat(pathFinal.c_str(), &statFinal) < 0) {
    // no such file, means no EOF yet
    return false;
  }

  // check current position
  off_t offset = ::lseek(m_fd, 0, SEEK_CUR);
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
  if (::fstat(m_fd, &statCurrent) < 0) {
    MsgLog(logger, error, "error returned from stat: " << errno << " -- " << strerror(errno));
    return false;
  }

  // compare inodes
  return statFinal.st_dev == statCurrent.st_dev and statFinal.st_ino == statCurrent.st_ino;
}

} // namespace XtcInput
