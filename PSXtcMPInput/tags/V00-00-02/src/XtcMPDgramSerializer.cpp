//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcMPDgramSerializer...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcMPInput/XtcMPDgramSerializer.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <cstring>
#include <stdint.h>
#include <boost/foreach.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSXtcMPInput/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char logger[] = "XtcMPDgramSerializer";
  
  const char HEADER[] = "DGRM";
  const char TRAILER[] = "MRGD";

  const size_t maxDataSize = 512*1024*1024;  // max expected data size (sanity check)

  // special delete for boost::shared_ptr
  struct CharArrayDeleter {
    void operator()(char* p) { delete [] p; }
  };

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSXtcMPInput {

//----------------
// Constructors --
//----------------
XtcMPDgramSerializer::XtcMPDgramSerializer (int fd)
  : m_fd(fd)
{
}

/// send all datagrams to a stream, throws exception in case of trouble.
void
XtcMPDgramSerializer::serialize(const std::vector<XtcInput::Dgram>& eventDg, const std::vector<XtcInput::Dgram>& nonEventDg)
{

  // calculate total size of the data
  uint32_t totalSize = 4 + 4; // HEADER and SIZE
  BOOST_FOREACH (const XtcInput::Dgram& dg, eventDg) {
    const std::string& path = dg.file().path();
    totalSize += 1 + path.size() + 1;
    totalSize = ((totalSize + 3) / 4) * 4;  // align for 4 bytes
    totalSize += dg.dg()->xtc.sizeofPayload() + sizeof(Pds::Dgram);
    totalSize = ((totalSize + 3) / 4) * 4;  // align for 4 bytes
  }
  BOOST_FOREACH (const XtcInput::Dgram& dg, nonEventDg) {
    const std::string& path = dg.file().path();
    totalSize += 1 + path.size() + 1;
    totalSize = ((totalSize + 3) / 4) * 4;  // align for 4 bytes
    totalSize += dg.dg()->xtc.sizeofPayload() + sizeof(Pds::Dgram);
    totalSize = ((totalSize + 3) / 4) * 4;  // align for 4 bytes
  }
  totalSize += 4; // TRAILER


  uint32_t off = 0;
  send(HEADER, 4);
  off += 4;
  send(&totalSize, 4);
  off += 4;

  BOOST_FOREACH (const XtcInput::Dgram& dg, eventDg) {
    off += sendDg(dg, 1);
  }

  BOOST_FOREACH (const XtcInput::Dgram& dg, nonEventDg) {
    off += sendDg(dg, 0);
  }

  send(TRAILER, 4);
  off += 4;

  MsgLog(logger, debug, "sent " << off << " bytes to fd " << m_fd << " with " << eventDg.size() << "+" << nonEventDg.size() << " datagrams");
}

// retrieve all datagrams from a stream, throws exception in case of trouble.
void 
XtcMPDgramSerializer::deserialize(std::vector<XtcInput::Dgram>& eventDg, std::vector<XtcInput::Dgram>& nonEventDg)
{
  // read header (first 8 bytes)
  char hdr[8];
  unsigned read = recv(hdr, 8);
  if (read != 8) {
    // master died in the middle of sending data
    throw ProtocolError(ERR_LOC, "master disconnected unexpectedly");
  }
  
  if (not std::equal(HEADER, HEADER+4, hdr)) {
    throw ProtocolError(ERR_LOC, "Unexpected header when receiving datagrams from master");
  }

  uint32_t totalSize = *(uint32_t*)(hdr+4);
  MsgLog(logger, debug, "total message size " << totalSize);
  if (totalSize > ::maxDataSize) {
    throw ProtocolError(ERR_LOC, "Unexpected size in header when receiving datagrams from master");
  }
  
  boost::shared_ptr<char> buf = boost::shared_ptr<char>(new char[totalSize-8], ::CharArrayDeleter());

  // read remaining message
  read = recv(buf.get(), totalSize-8);
  if (read != totalSize-8) {
    throw ProtocolError(ERR_LOC, "master disconnected unexpectedly");
  }

  // unpack all stuff
  unsigned off = 0;
  const char* const ptr = buf.get();
  while (off < totalSize-8-4) {
    
    // 1-byte flag
    uint8_t flag = ptr[off];
    off += 1;

    // file name
    size_t flen = std::strlen(ptr+off);
    std::string path(ptr+off, flen);
    off += flen+1;  // zero terminator

    // align
    off = ((off + 3) / 4) * 4;
    
    // datagram is next
    XtcInput::Dgram::ptr dgptr(buf, (Pds::Dgram*)(ptr + off));
    off += dgptr->xtc.sizeofPayload() + sizeof(Pds::Dgram);

    XtcInput::Dgram dg(dgptr, XtcInput::XtcFileName(path));
    if (flag) {
      eventDg.push_back(dg);
    } else {
      nonEventDg.push_back(dg);
    }

    // align
    off = ((off + 3) / 4) * 4;
  }
  
  // check trailer
  if (off != totalSize-8-4) {
    throw ProtocolError(ERR_LOC, "data packet internal structure is corrupted");
  }
  
  if (not std::equal(TRAILER, TRAILER+4, ptr+off)) {
    throw ProtocolError(ERR_LOC, "Unexpected trailer when receiving datagrams from master");
  }

  MsgLog(logger, debug, "received " << eventDg.size() << "+" << nonEventDg.size() << " datagrams");
}

void
XtcMPDgramSerializer::send(const void* buf, size_t size)
{

  const char* ptr = static_cast<const char*>(buf);
  while (size != 0) {

    ssize_t sent = ::write(m_fd, ptr, size);
    if (sent == 0) {
      // worker closed its connection?
      throw ProtocolError(ERR_LOC, "worker closed connection while master was sending data");
    }
    if (sent < 0) {
      if (errno == EINTR) {
        // retry
        sent = 0;
      } else {
        // error happened
        throw ExceptionErrno(ERR_LOC, "writing to a worker pipe failed");
      }
    }

    // advance
    ptr += sent;
    size -= sent;
  }

}

int
XtcMPDgramSerializer::sendDg(const XtcInput::Dgram& dg, uint8_t flag)
{
  int off = 0;
  send(&flag, 1);
  off += 1;

  const std::string& path = dg.file().path();
  send(path.c_str(), path.size() + 1);
  off += path.size() + 1;

  uint32_t pad = 0;
  size_t padsize = ((off + 3) / 4) * 4 - off;
  if (padsize) {
    send(&pad, padsize);
    off += padsize;
  }

  uint32_t dgsize = dg.dg()->xtc.sizeofPayload() + sizeof(Pds::Dgram);
  send(dg.dg().get(), dgsize);
  off += dgsize;

  padsize = ((off + 3) / 4) * 4 - off;
  if (padsize) {
    send(&pad, padsize);
    off += padsize;
  }

  return off;
}

int
XtcMPDgramSerializer::recv(void* buf, size_t size)
{

  char* ptr = static_cast<char*>(buf);
  size_t read = 0;
  while (read != size) {

    ssize_t received = ::read(m_fd, ptr, size - read);
    if (received == 0) {
      // EOF
      return read;
    }
    if (received < 0) {
      if (errno == EINTR) {
        // retry
        received = 0;
      } else {
        // error happened
        throw ExceptionErrno(ERR_LOC, "reading from data pipe failed");
      }
    } else {
      
    }

    // advance
    ptr += received;
    read += received;
  }

  return read;
}


} // namespace PSXtcMPInput
