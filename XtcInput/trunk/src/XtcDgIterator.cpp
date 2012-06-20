//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcDgIterator...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/XtcDgIterator.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/Exceptions.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "XtcDgIterator";

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
XtcDgIterator::XtcDgIterator (const std::string& path, size_t maxDgramSize)
  : m_path(path)
  , m_fd(-1)
  , m_maxDgramSize(maxDgramSize)
{
  m_fd = open( m_path.c_str(), O_RDONLY|O_LARGEFILE);
  if (m_fd < 0) {
    MsgLog( logger, error, "failed to open input XTC file: " << m_path ) ;
    throw FileOpenException(m_path) ;
  }
}

//--------------
// Destructor --
//--------------
XtcDgIterator::~XtcDgIterator ()
{
  if (m_fd >= 0) close(m_fd);
}

Dgram::ptr
XtcDgIterator::next()
{
  Pds::Dgram header;
  const size_t headerSize = sizeof(Pds::Dgram);

  // read header
  size_t left = headerSize;
  while (left > 0) {
    ssize_t nread = read(m_fd, ((char*)&header)+headerSize-left, left);
    if (nread == 0) {
      if (left != headerSize) {
        MsgLog(logger, error, "EOF while reading datagram header from file: " << m_path);
      }
      return Dgram::ptr();
    } else if (nread < 0) {
      if (errno == EINTR) continue;
      throw XTCReadException(m_path);
    } else {
      left -= nread;
    }
  }

  // check payload size, protection against corrupted headers
  size_t payloadSize = header.xtc.sizeofPayload();
  if (payloadSize > (m_maxDgramSize-headerSize)) {
    throw XTCSizeLimitException(m_path, payloadSize+headerSize, m_maxDgramSize);
  }

  Pds::Dgram* dg = (Pds::Dgram*)new char[payloadSize+headerSize];
  std::copy((const char*)&header, ((const char*)&header)+headerSize, (char*)dg);

  // read rest of the data
  left = payloadSize;
  while (left > 0) {
    ssize_t nread = read(m_fd, dg->xtc.payload()+payloadSize-left, left);
    if (nread == 0) {
      MsgLog(logger, error, "EOF while reading datagram payload from file: " << m_path);
      return Dgram::ptr();
    } else if (nread < 0) {
      if (errno == EINTR) continue;
      throw XTCReadException(m_path);
    } else {
      left -= nread;
    }
  }

  return Dgram::make_ptr(dg);
}

} // namespace XtcInput
