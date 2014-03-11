//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DgHeader...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/DgHeader.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/Exceptions.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "XtcInput.DgHeader";

  // absolute upper limit on the size of datagram that we can ever read
  // set it to 256MB for now
  size_t maxDgramSize = 256*1024*1024;

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
DgHeader::DgHeader(const Pds::Dgram& header, const SharedFile& file, off_t off)
  : m_header()
  , m_file(file)
  , m_off(off)
{
  // Dgram copy constructor does not work like we need, do byte-copy instead for sure way
  std::copy((const char*)&header, ((const char*)&header)+sizeof header, (char*)&m_header);
}

/// Returns offset of the next header (if there is any)
off_t
DgHeader::nextOffset() const
{
  return m_off + (sizeof m_header + m_header.xtc.extent - sizeof m_header.xtc);
}

/// Reads complete datagram into memory
Dgram::ptr
DgHeader::dgram()
{
  const size_t headerSize = sizeof m_header;
  const uint32_t payloadSize = m_header.xtc.extent - sizeof m_header.xtc;
  const uint32_t datagramSize = headerSize + payloadSize;

  // check datagram size, protection against corrupted headers
  MsgLog(logger, debug, "XTC extent size = " << m_header.xtc.extent);
  if (datagramSize > ::maxDgramSize) {
    throw XTCSizeLimitException(ERR_LOC, m_file.path().path(), datagramSize, ::maxDgramSize);
  }

  // allocate memory for header+payload
  Pds::Dgram* dg = (Pds::Dgram*)new char[datagramSize];

  // copy header
  std::copy((const char*)&m_header, ((const char*)&m_header)+headerSize, (char*)dg);

  // wrap into smart pointer so it gets deleted
  Dgram::ptr dgram = Dgram::make_ptr(dg);

  // make sure that we are at correct location
  m_file.seek(m_off + headerSize, SEEK_SET);

  // read rest of the data
  MsgLog(logger, debug, "reading payload, size = " << payloadSize << ", offset = " << m_off);
  ssize_t nread = m_file.read(dg->xtc.payload(), payloadSize);
  if (nread < 0) {
    throw XTCReadException(ERR_LOC, m_file.path().path());
  } else if (nread != ssize_t(payloadSize)) {
    MsgLog(logger, warning, "EOF while reading datagram payload from file: " << m_file.path());
    return Dgram::ptr();
  }

  return dgram;
}

} // namespace XtcInput
