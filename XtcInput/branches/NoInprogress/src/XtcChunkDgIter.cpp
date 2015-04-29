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
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/Exceptions.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "XtcInput.XtcChunkDgIter";

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
XtcChunkDgIter::XtcChunkDgIter (const XtcFileName& path, unsigned liveTimeout)
  : m_file(path, liveTimeout)
  , m_off(0)
{
}

//--------------
// Destructor --
//--------------
XtcChunkDgIter::~XtcChunkDgIter ()
{
}

boost::shared_ptr<DgHeader>
XtcChunkDgIter::next()
{
  return nextAtOffset(m_off);
}

boost::shared_ptr<DgHeader>
XtcChunkDgIter::nextAtOffset(off64_t offset)
{
  if (m_file.seek(offset, SEEK_SET) == (off_t)-1) {
    throw XTCReadException(ERR_LOC, m_file.path().path());
  }

  boost::shared_ptr<DgHeader> hptr;

  // read header
  Pds::Dgram header;
  const size_t headerSize = sizeof header;
  MsgLog(logger, debug, "reading header");
  ssize_t nread = m_file.read(((char*)&header), headerSize);
  if (nread == 0) {
    // EOF
    return hptr;
  } else if (nread < 0) {
    throw XTCReadException(ERR_LOC, m_file.path().path());
  } else if (nread != ssize_t(headerSize)) {
    MsgLog(logger, error, "EOF while reading datagram header from file: " << m_file.path());
    return hptr;
  }

  WithMsgLog(logger, debug, str) {
    str << "header:";
    uint32_t* p = (uint32_t*)&header;
    for (int i = 0; i != 10; ++ i) str << ' ' << p[i];
  }

  // check header consistency
  if (header.xtc.extent < sizeof(Pds::Xtc)) {
    throw XTCExtentException(ERR_LOC, m_file.path().path(), offset, header.xtc.extent);
  }

  // make an object
  hptr = boost::make_shared<DgHeader>(header, m_file, offset);

  // get position of the next datagram
  m_off = hptr->nextOffset();

  return hptr;
}

} // namespace XtcInput
